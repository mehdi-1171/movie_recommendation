from pandas import read_csv, DataFrame
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, GridSearchCV
from time import time
import pickle

from utility import DATA_PATH


class DevelopModel:
    MOVIES_FILE = 'movies.csv'
    RATINGS_FILES = 'ratings.csv'
    MODEL_PATH = DATA_PATH + 'movieLens_svd_model.pkl'

    def __init__(self):
        self.df_movie = DataFrame()
        self.df_rating = DataFrame()
        self.df = DataFrame()
        self.data_set = DataFrame()
        self.model = None
        self.train_set = None
        self.test_set = None

    def read_movie_data(self):
        self.df_movie = read_csv(DATA_PATH + self.MOVIES_FILE)

    def read_rating_data(self):
        self.df_rating = read_csv(DATA_PATH + self.RATINGS_FILES)

    def prepare_data(self):
        reader = Reader(rating_scale=(1, 5))
        self.data_set = Dataset.load_from_df(
            self.df_rating[['userId', 'movieId', 'rating']],
            reader
        )

    def separate_train_test_data(self):
        self.train_set, self.test_set = train_test_split(
            self.data_set, test_size=0.2, random_state=42
        )

    def grid_search_svd(self):
        param_grid = {
            'n_factors': [100, 150, 200],
            'n_epochs': [20, 30],
            'lr_all': [0.002, 0.005],
            'reg_all': [0.02, 0.05]
        }

        print("🔍 Starting GridSearch for SVD ...")
        start_time = time()

        gs = GridSearchCV(
            SVD,
            param_grid,
            measures=['rmse'],
            cv=3,
            n_jobs=-1,
            joblib_verbose=1
        )

        gs.fit(self.data_set)

        elapsed = time() - start_time
        print(f"✅ GridSearch finished in {elapsed:.2f} seconds")

        print("🏆 Best RMSE:", gs.best_score['rmse'])
        print("⚙️ Best Params:", gs.best_params['rmse'])

        self.model = gs.best_estimator['rmse']

    def train_best_model(self):
        if self.model is None:
            raise ValueError("Model is not defined. Run grid_search_svd first.")

        print("🚀 Training best SVD model on train set...")
        self.model.fit(self.train_set)

    def evaluate_model(self):
        if self.model is None:
            raise ValueError("Model is not trained.")

        predictions = self.model.test(self.test_set)
        rmse = accuracy.rmse(predictions, verbose=True)
        print(f'Evaluate Model on test Data and give RMSE: {rmse} ')

    def save_model(self):
        if self.model is None:
            raise ValueError("No model to save.")

        with open(self.MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)

        print(f"💾 Model saved successfully at: {self.MODEL_PATH}")

    def process_handler(self):
        self.read_movie_data()
        self.read_rating_data()
        self.prepare_data()
        self.separate_train_test_data()
        self.grid_search_svd()
        self.train_best_model()
        self.evaluate_model()
        self.save_model()


""" TEST """
# dd = DevelopModel()
# dd.process_handler()
