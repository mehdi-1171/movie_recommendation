from pandas import read_csv, DataFrame
import pickle

from utility import DATA_PATH


class RecommendationEngine:
    MOVIES_FILE = 'movies.csv'
    RATINGS_FILE = 'ratings.csv'
    MODEL_NAME = 'movieLens_svd_model.pkl'

    def __init__(self, user_id, top_n=5):
        self.user_id = user_id
        self.top_n = top_n
        self.model = None
        self.df_movies = DataFrame()
        self.df_ratings = DataFrame()
        self.rated_movie = DataFrame()
        self.unseen_movies = DataFrame()
        self.predict_df = DataFrame()
        self.top_df = None

    def top_movie_getter(self):
        self.recommended_process()
        return self.top_df

    def load_model(self):
        with open(DATA_PATH + self.MODEL_NAME, 'rb') as f:
            self.model = pickle.load(f)

    def load_data(self):
        self.df_movies = read_csv(DATA_PATH + self.MOVIES_FILE)
        self.df_ratings = read_csv(DATA_PATH + self.RATINGS_FILE)

    def get_unseen_movies(self):
        self.rated_movie = self.df_ratings[
            self.df_ratings['userId'] == self.user_id
        ]
        print(self.rated_movie['rating'].mean())
        rated_movie_ids = self.rated_movie['movieId'].unique()
        self.unseen_movies = self.df_movies[
            ~self.df_movies['movieId'].isin(rated_movie_ids)
        ]

    def predict_ratings(self):
        predictions = []
        for movie_id in self.unseen_movies['movieId']:
            pred = self.model.predict(self.user_id, movie_id)
            predictions.append((movie_id, pred.est))
        self.predict_df = DataFrame(predictions, columns=['movieId', 'predicted_rating'])

    def top_n_best(self):
        self.top_df = (self.predict_df.sort_values(by='predicted_rating', ascending=False).head(self.top_n))

    def attach_movie_info(self):
        self.top_df = self.top_df.merge(self.df_movies,
                                        on='movieId',
                                        how='left')[['movieId', 'title', 'genres', 'predicted_rating']]

    def recommended_process(self):
        self.load_model()
        self.load_data()
        self.get_unseen_movies()
        self.predict_ratings()
        self.top_n_best()
        self.attach_movie_info()


""" TEST """
# r = RecommendationEngine(user_id=4, top_n=5)
# result_df = r.top_movie_getter()
# print(result_df.to_markdown())

# r.load_model()
# r.load_data()
# r.get_unseen_movies()
# r.predict_ratings()
# r.top_n_best()
# r.attach_movie_info()
