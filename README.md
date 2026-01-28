# Movie Recommendation System

## Objective
The objective of this project is to build a robust and scalable movie recommendation system using the MovieLens dataset. 
The system employs collaborative filtering techniques to predict user ratings and recommend movies that users are likely to enjoy. 
The project is divided into two main phases: 
Model Development and Recommendation Engine, with a focus on achieving optimal 
RMSE (Root Mean Square Error) while maintaining practical usability.

## 🤖 Recommendation System

>                               -------------------------
>                               | Recommendation System |
>                               -------------------------
>                                           |
>               ----------------------------|----------------------------
>               |                           |                           |  
>        ----------------       ---------------------------      -------------- 
>       | Content Based |      | Collaborative Filtering  |      |   Hybrid   | 
>        ----------------       ---------------------------      --------------
>                                           |
>                          -------------------------------------
>                         |                 |                  |
>                   -------------     -------------      ------------
>                   | Item-Item |     | Item-User |      | User-User |
>                   -------------     -------------      ------------- 


## Project Structure

### 📊 Phase 1: Model Development (Training)
The first phase focuses on developing and optimizing the recommendation model:

1. Select Recommendation System Approach
   - Collaborative Filtering (User-Item Matrix Factorization)
   - Chosen for its effectiveness with explicit feedback (ratings
2. Prepare Data for Recommendation System
   - Load MovieLens dataset (movies.csv, ratings.csv)
   - Filter out users and movies with insufficient ratings
   - Transform data into Surprise library format
3. Algorithm Selection & Comparison
   - Compare multiple algorithms: SVD, SVD++, NMF, SlopeOne, CoClustering, BaselineOnly, KNNBaseline
   - Use cross-validation (3-5 folds) for fair comparison
   - Evaluate based on RMSE, MAE, and training time

4. Model Training & Validation
   - Split data into training (80%) and testing (20%) sets
   - Train selected algorithm on training set
   - Validate using test set to prevent overfitting

5. Hyperparameter Tuning
   - Use GridSearchCV for systematic hyperparameter optimization
   - Tune: n_factors, n_epochs, learning rate, regularization parameters
   - Select parameters that minimize RMSE while preventing overfitting
6. Model Persistence
   - Save trained model with metadata (training date, parameters, performance metrics)
   - Enable model reuse without retraining

### 🚀 Phase 2: Recommendation Engine
The second phase focuses on generating personalized recommendations:

1. Load Pre-trained Model
   - Load saved model and metadata
   - Verify model integrity and performance metrics
2. Data Loading & Preparation
   - Load movie and rating data
   - Prepare user-specific data structures
3. Identify Unseen Movies
   - For each user, identify movies they haven't rated
   - Filter out movies with insufficient global ratings if needed
4. Rating Prediction
   - Use trained model to predict ratings for unseen movies
   - Generate estimated ratings for all candidate movies
5. Top-N Recommendations Generation
   - Sort predicted ratings in descending order
   - Select top N movies as recommendations
   - Combine with movie metadata for meaningful output
6. Result Presentation
   - Display recommendations with movie titles, genres, and predicted ratings
   - Provide confidence scores for recommendations
 
### 🌐 Phase 3: Service Development (Optional)
Future development to deploy the system as a service:

- RESTful API with FastAPI
- Web interface with Streamlit
- Real-time recommendation generation
- Scalable deployment options

## 📁 Dataset
MovieLens Dataset - A benchmark dataset for recommendation systems:

- Ratings: 100,000 ratings (1-5 scale) from 943 users on 1682 movies
- Movies: Movie metadata including title and genres
- Users: Anonymous user IDs only
- Timestamp: When the rating was given (not used in this implementation)

Key Statistics:

- Average ratings per user: ~106
- Average ratings per movie: ~59
- Rating density: ~6.3% (sparse matrix)
- Rating distribution: Approximately normal with mean ~3.5

## Exploratory Data Analysis

1. Rating
[Rating Distribution](https://github.com/mehdi-1171/movie_recommendation/blob/main/img/rating_distribution.png)
2. Some Key Result

|   |         Variable | value | 
|--:|-----------------:|:------|
| 1 |         Num User | 6781  |
| 2 |        Num Movie | 27922 |
| 3 |    Data sparsity | 0.55% |
| 4 |       Avg Rating | 154.6 |
| 5 | User Less Rating | 20    |
| 6 | User Most Rating | 3893  |


## 💻 Code Structure

      movie_recommendation/
      ├── data/
      │   ├── movies.csv                  # Movie metadata
      │   └── ratings.csv                 # User ratings
      │   └── movieLens_svd_model.pkl     # Train Model
      │
      ├── eda.py                          # Exploratory Data Analysis
      ├── recommendation_system.py        # Recommendation Engine module
      ├── train_model.py                  # Train model module
      ├── utility.py                      # utility of project
      ├── requirements.txt                # Python dependencies
      └── README.md                       # This file

## 🤖 Model Selection
### Why SVD (FunkSVD)?
After comprehensive comparison of multiple algorithms, SVD (Singular Value Decomposition) was selected as the primary model due to:

1- Performance: Achieves RMSE of 0.82-0.84, which is competitive with more complex models
2- Efficiency: Faster training and prediction compared to SVD++
3- Interpretability: Latent factors can be interpreted as abstract features
4- Robustness: Less prone to overfitting with proper regularization
5- Scalability: Can handle large datasets efficiently

### Optimal Hyperparameters (for MovieLens 100K):

- Best Hyperparameter

> - **n_factors:** 200 (latent features)
> - **n_epochs:** 150 (training iterations)
> - **lr_all:** 0.005 (learning rate)
> - **reg_all:** 0.05 (regularization strength)
> - **random_state:** 42 (for reproducibility)

## 📊 Evaluation Methodology

### Metrics Used:
1. **RMSE (Root Mean Square Error):** Primary metric, measures average magnitude of errors
2. **MAE (Mean Absolute Error):** Alternative error metric less sensitive to outliers
3. **Cross-Validation:** 3-5 fold CV to ensure robustness
4. **Training Time:** Practical consideration for real-world deployment

### Validation Strategy:
- **Train/Test Split:** 80/20 split with random state for reproducibility
- **Stratified Sampling:** Ensure all users are represented in both sets
- **Cold-Start Handling:** Users with <20 ratings filtered out
- **Item Popularity:** Movies with <10 ratings filtered out

## 📈 Comparison Model
- Although SVD++ achieves slightly lower RMSE, SVD was selected due to its significantly faster training time and comparable performance.

|   |    Model | RMSE | Run Time  |
|--:|---------:|:-----|:----------|
| 1 | Baseline | 0.94 | Very Fast |
| 2 |    SVD   | 0.82 | Fast      |
| 3 |    SVD++ | 0.81 | Very Slow |



## 🎬 Recommendation Results
For example i recommended 5 best movies for **user_id = 4** :

  |   | movieId | title                           | genres              | predicted_rating |
  |--:|--------:|:--------------------------------|:--------------------|-----------------:|
  | 1 |     665 | Underground (1995)              | Comedy-Drama-War    |          3.78545 |
  | 2 |  198185 | Twin Peaks (1989)               | Drama-Mystery       |          3.77980 |
  | 3 |   26587 | Decalogue, The (Dekalog) (1989) | Crime-Drama-Romance |          3.77533 |
  | 4 |  171011 | Planet Earth II (2016)          | Documentary         |          3.76678 |
  | 5 |  151113 | A Man Called Ove (2015)         | Comedy-Drama        |          3.75257 |

Interpretation: The system recommends diverse genres including drama, documentary, and comedy, with predicted ratings all above 3.75 (indicating strong recommendations).

## 🎯 Conclusion

### Key Achievements:
- **Optimized Model:** Using tuned SVD, we achieved an RMSE of 0.82 on MovieLens via cross-validation
- **Practical Performance:** An RMSE of 0.82 indicates that predicted ratings deviate from true ratings by less than one star on average
- **Efficient Implementation:** The model trains in under a minute while maintaining high accuracy
- **Scalable Design:** Modular architecture allows easy extension to other datasets or algorithms

### Performance Interpretation:
- **RMSE = 0.82:** This is considered strong performance for MovieLens 100K dataset
- **Industry Standard:** Comparable to published results in academic literature
- **Practical Utility:** Provides meaningful recommendations that align with user preferences

### Future Improvements:
1. **Hybrid Approach:** Combine collaborative filtering with content-based features
2. **Deep Learning:** Experiment with neural collaborative filtering
3. **Real-time Updates:** Implement incremental learning for new ratings
4. **Explainability:** Add feature to explain why movies are recommended

## 👨‍💻 Author

This Project Develop by [Mehdi](m.habibian1171@gmail.com) If you have a question don't be hesitated

- Last Updated: January 2026
- Version: 1.0
- Status: Active Development

## 📄Licence
This project uses the MovieLens 100K dataset provided by GroupLens Research.
The dataset is licensed for non-commercial research and educational purposes only.

For full license terms, see: https://files.grouplens.org/datasets/movielens/ml-100k-README.txt
