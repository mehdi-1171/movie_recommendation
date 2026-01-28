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
        """تحلیل داده‌ها"""
        print("\n=== تحلیل داده‌ها ===")

        # توزیع ریتینگ‌ها
        rating_dist = self.df_rating['rating'].value_counts().sort_index()
        print("توزیع ریتینگ‌ها:")
        for rating, count in rating_dist.items():
            print(f"  {rating}: {count} ({count / len(self.df_rating) * 100:.1f}%)")

        # تعداد کاربران و فیلم‌ها
        n_users = self.df_rating['userId'].nunique()
        n_movies = self.df_rating['movieId'].nunique()
        print(f"\nتعداد کاربران: {n_users}")
        print(f"تعداد فیلم‌ها: {n_movies}")

        # تراکم ماتریس
        density = len(self.df_rating) / (n_users * n_movies) * 100
        print(f"تراکم داده‌ها: {density:.4f}%")

        # فعالیت کاربران
        user_activity = self.df_rating['userId'].value_counts()
        print(f"\nمیانگین ریتینگ هر کاربر: {user_activity.mean():.1f}")
        print(f"کاربر با کمترین ریتینگ: {user_activity.min()}")
        print(f"کاربر با بیشترین ریتینگ: {user_activity.max()}")

    def analysis_process(self):
        self.read_movie_data()
        self.read_rating_data()
        self.analyze_data()


""" TEST """
# a = DataAnalysis()
# a.analysis_process()