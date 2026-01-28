from pandas import read_csv, merge, DataFrame
import os

from utility import DATA_PATH


class DataAnalysis:
    MOVIES_FILE = 'movies.csv'
    RATINGS_FILES = 'ratings.csv'

    def __init__(self):
        self.df_movie = DataFrame()
        self.df_rating = DataFrame()
        self.df = DataFrame()

    def read_movie_data(self):
        self.df_movie = read_csv(DATA_PATH + self.MOVIES_FILE)

    def read_rating_data(self):
        self.df_rating = read_csv(DATA_PATH + self.RATINGS_FILES)

    def analyze_data(self):
        print("\n=== Data Analysis ===")

        rating_dist = self.df_rating['rating'].value_counts().sort_index()
        print("Rating Distribution:")
        for rating, count in rating_dist.items():
            print(f"  {rating}: {count} ({count / len(self.df_rating) * 100:.1f}%)")

        n_users = self.df_rating['userId'].nunique()
        n_movies = self.df_rating['movieId'].nunique()
        print(f"\nNumber of User: {n_users}")
        print(f"Number of Movies {n_movies}")

        density = len(self.df_rating) / (n_users * n_movies) * 100
        print(f"Data sparsity: {density:.4f}%")

        user_activity = self.df_rating['userId'].value_counts()
        print(f"\nAvg of any user rating: {user_activity.mean():.1f}")
        print(f"A user with less rating  {user_activity.min()}")
        print(f"A User with most rating {user_activity.max()}")

    def analysis_process(self):
        self.read_movie_data()
        self.read_rating_data()
        self.analyze_data()


""" TEST """
# a = DataAnalysis()
# a.analysis_process()