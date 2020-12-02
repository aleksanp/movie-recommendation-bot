# !pip install lightfm
# !pip install pytelegrambotapi

import numpy as np
import pandas as pd
import telebot
import datetime

from lightfm import LightFM
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from telebot import types


class MovieRecommender():
    
    """
    Telegram chat-bot which can recommend films and save user scores
    """
    
    def __init__(self, dataframe_movies, dataframe_ratings):
        
        self.data_movies = dataframe_movies.copy()
        self.data_ratings = dataframe_ratings.copy()

        self.user_id_col = 'userId' 
        self.movie_id_col = 'movieId'
        self.rating_col = 'rating'
        self.title_col = 'title'
        
        self.rating_output_filename = 'df_ratings.csv'
        
        # Default values
        self.model_parameters = {
            'no_components': 30,
            'loss': 'warp',
            'k': 15
        }
        self.fit_parameters = {
            'epochs': 30,
            'num_threads': 4
        }
        self.threshold = 4.0
        self.n_rec_items = 6
        
        self.user_id_ = 0
        self.selected_movie_id_ = 0
        self.interactions_ = None
        self.similarity_matrix_ = None
        self.model_ = None
        self.users_df_ = None
        self.movies_df_ = None
        self.mean_total_score = None
        
        
    def learn(self, model_kwargs=None, fit_kwargs=None):
        
        if not model_kwargs:
            model_kwargs = self.model_parameters
        if not fit_kwargs:
            fit_kwargs = self.fit_parameters
        
        self.interactions_ = self.get_interaction_matrix()
        self.users_df_ = pd.DataFrame(self.get_interaction_matrix().index)
        self.movies_df_ = (self.data_movies[[self.movie_id_col, self.title_col]]
                              .set_index([self.movie_id_col]))
        self.model_ = self.prediction_model(model_kwargs, fit_kwargs)
        self.similarity_matrix_ = self.get_items_similarity_matrix()
        
        return self.model_
    
    
    def run(self):
        
        """
        It runs the telegram chat-bot. You must use your own api key.
        The bot can recommend N movies, save new scores and relearn.
        """
         
        with open ('api_key_for_bot.txt') as api_key_file:
            api_key = api_key_file.read().strip()
        
        bot = telebot.TeleBot(api_key)     
        
        @bot.message_handler(content_types=['text'])
        def get_text_messages(message):
            
            self.user_id_ = message.from_user.id

            # Possible scores 0.5 , 1.0 or 1, ... 5 or 5.0 
            possible_scores = (list(map(str, np.arange(0.5, 5.1, 0.5))) 
                               + list(map(str, np.arange(1, 6))))
                
            if message.text.lower() in ['привет', 'подскажи', 'далее', 'ещё', 'еще']:
                keyboard = types.InlineKeyboardMarkup()
                recommendations = self.get_recommendation_for_user()[self.title_col].items()

                for movie_id, movie_name in recommendations:
                    key = types.InlineKeyboardButton(text=movie_name, callback_data=movie_id)
                    keyboard.add(key)
                    
                if message.text.lower() in ['привет']:
                    question = "Привет, а эти фильмы смотрел?"
                else:
                    question = "А эти фильмы смотрел?"
                    
                bot.send_message(message.from_user.id, text=question, reply_markup=keyboard)
                bot.send_message(message.from_user.id, 
                    "Выбери фильм и оцени его (0.5, 1, ... 5)")
            
            elif message.text in possible_scores and self.selected_movie_id_:
                self.save_score(self.selected_movie_id_, float(message.text), 
                                filename=self.rating_output_filename)
                print('data ratings:\n', self.data_ratings.tail(1))
                # Reset the movie id
                self.selected_movie_id_ = 0
               
            elif message.text.lower() in ['запомнить', 'сохранить']:
                # Relearn the model with new datas
                self.data_ratings = pd.read_csv(self.rating_output_filename)
                self.learn()
                
            elif message.text == "/help":
                bot.send_message(message.from_user.id, 
                    'Напиши привет или ещё, чтобы получить рекомендацию. '
                    'Далее выбери фильм и введи оценку (0.5 - 5.0), если уже смотрел(а) его. '
                    'Можно оценить все из предложенных фильмов по очереди. '
                    'Далее чат-бот можно переобучить на основе ваших интересов. ' 
                    'Для этого после введённых оценок напишите сохранить или переобучить') 
                
            else:
                bot.send_message(message.from_user.id, 
                    "Я тебя не понимаю. Напиши /help, привет или ещё.")   
            
        @bot.callback_query_handler(func=lambda call: True)
        def select_movie(call):
            self.selected_movie_id_ = call.data      
            print('selected_movie_id:', self.selected_movie_id_)

        # Run the bot
        bot.polling(none_stop=True, interval=0)
        
        
    def get_score(scores, threshold, mean_total_score):

        """
        Formula for calculating the movie score
            (V / (V+M)) * R + (M / (V+M)) * C
        V - number of votes
        M - threshold 
        R - average score for the movie среднее
        С - total average score for all movies
        """

        num_votes = len(scores)
        mean_movie_score = np.mean(scores)
        movie_score = (
            (num_votes / (num_votes + threshold)) * mean_movie_score +
            (threshold / (num_votes + threshold)) * mean_total_score)
        return movie_score
    
    
    def get_movie_scores(df_movies, df_ratings, threshold, mean_total_score):
    
        """
        Calculate the scores for all movies
        """
        
        movie_scores = []
        for index, row in df_movies.iterrows():
            movie_id = row[self.movie_id_col]
            scores = (df_ratings[df_ratings[self.movie_id_col] == movie_id]
                      [self.rating_col].to_list())
            movie_score = get_score(scores, threshold, mean_total_score)
            movie_scores.append(movie_score)
        movie_scores = pd.Series(movie_scores)
        movie_scores.name = 'score'
        return movie_scores


    def get_interaction_matrix(self, binary=False, threshold=None):
        
        """
        Return the movie-user interaction matrix, where the cells take the rating values
        """
        
        if not threshold:
            threshold = self.threshold
        
        interaction_matrix = (self.data_ratings
                              .groupby([self.user_id_col, self.movie_id_col])[self.rating_col]
                              .sum().unstack().reset_index()
                              .fillna(0).set_index(self.user_id_col))
        if binary:
            interaction_matrix = (interaction_matrix.applymap(
                lambda x: 1 if x > threshold else 0))
        return interaction_matrix
    
    
    def prediction_model(self, model_kwargs, fit_kwargs):

        """
        Learning the prediction model with a sparsed interaction matrix
        """

        x_train = sparse.csr_matrix(self.interactions_.values)
        model = LightFM(**model_kwargs)
        model.fit(x_train, **fit_kwargs)
        return model
    

    def get_recommendation_for_user(self, show=False):
        
        """
        The model predicts the movie ratings for current user. 
        Next it returns N top movies for current user
        """
        
        # Cold start with a new user        
        if not self.user_id_ in self.users_df_.values:
            user_id = self.users_df_.sample(1).values[0, 0]
        else:
            user_id = self.user_id_
            
        print('User ID =', user_id)
        
        n_users, n_items = self.interactions_.shape
        user_index = self.users_df_[self.users_df_ == user_id].index[0]
        scores = pd.Series(
            self.model_.predict(user_ids=user_index, item_ids=np.arange(n_items)))
        scores.index = self.interactions_.columns
        
        rated_movies = (self.interactions_.loc[user_id]
                        [self.interactions_.loc[user_id] > 0]
                        .sort_values(ascending=False))
        recommend_ids = (scores[~(self.interactions_.loc[user_id, :] > 0)]
                         .sort_values(ascending=False)
                         [:self.n_rec_items])
        
        rated = self.movies_df_.loc[rated_movies.index]
        recommedations = self.movies_df_.loc[recommend_ids.index]
        
        if show:
            print('Top watched \n')
            for value in rated[:self.n_rec_items].values:
                print('\t', value[0])
            print('\nRecommedations \n')
            for value in recommedations[:self.n_rec_items].values:
                print('\t', value[0]) 
                
        return recommedations
    
    
    def get_similar_users(self, number_of_user=10):
        
        """
        Return N users with similar interests
        """
        
        favorite_movies = (self.data_ratings[self.data_ratings[self.user_id_col] == self.user_id_]
                           .sort_values(by=self.rating_col, ascending=False)[self.movie_id_col]
                           .head(10))
        
        # !TO DO: Choose all movies
        if favorite_movies.empty:
            movie_id = 1
        else:
            movie_id = np.random.choice(favorite_movies.values)
            
        n_users, n_items = self.interactions_.shape
        movie_ids = np.array(self.interactions_.columns)
        scores = pd.Series(
            self.model_.predict(np.arange(n_users), 
                          np.repeat(movie_ids.searchsorted(movie_id), n_users)))
        similar_users = (scores.sort_values(ascending=False)[:number_of_user]
                         .index.to_list())
        return similar_users
    
    
    def get_items_similarity_matrix(self):
    
        """
        Return the movie-movie similarity matrix
        """
        
        similarity_matrix = pd.DataFrame(
            cosine_similarity(sparse.csr_matrix(self.model_.item_embeddings)))
        similarity_matrix.columns = self.interactions_.columns
        similarity_matrix.index = self.interactions_.columns
        
        return similarity_matrix
    
    
    def get_item_item_recommendation(self, movie_id, n_items=None, show=False):
        
        """
        Return the movie-movie recommendation based on the similarity matrix
        """
        if not n_items:
            n_items = self.n_rec_items
        recommended_movies = (self.similarity_matrix_
                              .loc[movie_id, :]
                              .sort_values(ascending=False)
                              [1: n_items+1])
        recommendation = self.movies_df_.loc[recommended_movies.index]
        
        if show:
            print(f'Recommendations:\n')
            for value in recommendation.values:
                print('\t', value[0])
                
        return recommendation

            
    def save_score(self, movie_id, score, by_name=False, filename=None):
        
        """
        The function saves new scores of the current user. 
        It can save due the run or into a file.
        In case when movie_id is a title use by_name=True
        """
        
        if by_name:
            
            movie_id = self.movies_df_[
                (self.movies_df_.iloc[:, 0] == movie_id)].index[0]
            
        timestamp = int(datetime.datetime.now().timestamp())
        
        self.data_ratings.loc[self.data_ratings.shape[0]] = [
            self.user_id_, movie_id, score, timestamp]
        
        if filename:
            self.data_ratings.to_csv(filename, index=False)       

        return self.data_ratings


def main():
    
    """Run bot."""
    
    data_movies = pd.read_csv('data/df_movies.csv', index_col=[0])
    data_ratings = pd.read_csv('data/df_ratings.csv', index_col=[0])

    movie_recommender_bot = MovieRecommender(data_movies, data_ratings)

    # Learn with default parameters
    movie_recommender_bot.learn()

    print('Run the bot...')
    movie_recommender_bot.run()

    # print(movie_recommender_bot.get_recommendation_for_user())


if __name__ == '__main__':
    main()
