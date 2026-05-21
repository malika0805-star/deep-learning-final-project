# Final Report

## Project Title

Movie Rating Prediction and Recommendation Using Neural Collaborative Filtering

---

## 1. Problem Statement

Online movie platforms contain thousands of movies, and users often cannot manually search through all available content. Because of this, recommendation systems are important tools for helping users find movies that match their interests.

The main problem of this project is to predict how a user would rate a specific movie. Based on predicted ratings, the system can rank movies and recommend the most relevant ones to each user.

This project is a **recommendation / ranking** task. It is not a sentiment classification task. The model does not classify reviews as positive or negative. Instead, it predicts a numerical rating value.

### Input

```text
userId
movieId
```

### Output

```text
predicted rating
```

The predicted rating represents the expected preference of a user for a movie. Higher predicted ratings indicate stronger recommendation relevance.

---

## 2. Dataset Description

The dataset used in this project is **MovieLens 20M**, a large movie rating dataset created by GroupLens. It is widely used for recommender system research and machine learning experiments.

### Dataset Source

```text
https://grouplens.org/datasets/movielens/20m/
```

### Direct Download Link

```text
https://files.grouplens.org/datasets/movielens/ml-20m.zip
```

### Dataset Size

The dataset contains:

```text
20,000,263 ratings
138,493 users
27,278 movies
```

The rating scale is:

```text
0.5 to 5.0
```

The main files used in this project are:

```text
ratings.csv
movies.csv
```

The `ratings.csv` file contains user-movie interactions. It includes `userId`, `movieId`, `rating`, and `timestamp`. The `movies.csv` file contains movie information such as `movieId`, `title`, and `genres`.

This dataset is suitable for the project because it directly represents the recommendation problem:

```text
user → movie → rating
```

This structure allows the model to learn user preferences and movie patterns from historical rating behavior.

---

## 3. Data Preprocessing

Data preprocessing was used to prepare the raw MovieLens data for machine learning and deep learning models.

The main input features were:

```text
userId
movieId
```

The prediction target was:

```text
rating
```

The original `userId` and `movieId` values were converted into numerical indexes:

```text
user_index
movie_index
```

This encoding was necessary because the deep learning model uses embedding layers. Embedding layers require integer indexes, so each user and movie must be represented by a continuous numerical ID starting from 0.

The dataset was divided into three parts:

| Dataset Part | Percentage | Purpose |
|---|---:|---|
| Train | 70% | Used to train the model |
| Validation | 15% | Used to monitor model performance |
| Test | 15% | Used for final evaluation |

This split makes the evaluation more reliable because the model is tested on data that was not used during training.

During preprocessing, missing values were checked, important columns were selected, and user/movie mappings were created. These mappings allow the model to connect original IDs with encoded indexes.

---

## 4. Model Architecture and Methods

This project used two main types of methods to solve the movie recommendation problem:

```text
1. Baseline average-based models
2. Neural Collaborative Filtering deep learning model
```

The baseline models were used as simple comparison methods. The deep learning model was used to learn more complex user-movie interaction patterns.

---

## 4.1 Baseline Models

Baseline models are simple models that help evaluate whether a more complex model is actually useful. In this project, the baseline models predict ratings using average rating values.

The main idea of baseline models is that useful information can already be found in the rating history. For example, some movies usually receive high ratings, while some users usually give higher or lower ratings than others.

The following baseline models were implemented:

```text
Global Average Rating
Movie Average Rating
User Average Rating
Combined User-Movie Average Rating
```

---

### Global Average Rating Baseline

The Global Average baseline is the simplest model. It calculates the average rating from the training dataset and predicts this same value for every user-movie pair.

For example, if the average rating in the training data is 3.5, then the model predicts:

```text
predicted rating = 3.5
```

for every user and every movie.

This model does not use personal user preferences or movie-specific information. It only gives a general average prediction. Because of this, it is useful as the lowest-level comparison model.

---

### Movie Average Rating Baseline

The Movie Average baseline predicts ratings using the average rating of each movie in the training data.

The idea is that some movies are generally liked more than others. If a movie usually receives high ratings from many users, the model predicts a higher rating for that movie.

For example:

```text
Movie A average rating = 4.3
Movie B average rating = 2.8
```

Then the model predicts a higher rating for Movie A because it has stronger historical feedback.

If a movie from the test set does not exist in the training set, the model uses the global average rating as a fallback value.

This method is stronger than the global average because it uses movie popularity and quality information.

---

### User Average Rating Baseline

The User Average baseline predicts ratings based on each user’s average rating behavior.

The idea is that users have different rating habits. Some users usually give high ratings, while others are stricter and give lower ratings.

For example:

```text
User A average rating = 4.2
User B average rating = 2.9
```

If User A usually gives high ratings, the model predicts a higher rating for movies rated by User A.

If a user from the test set does not exist in the training set, the model uses the global average rating.

This method is useful because it considers individual user behavior, but it does not directly consider whether a specific movie is popular or unpopular.

---

### Combined User-Movie Average Baseline

The Combined User-Movie Average baseline combines both user behavior and movie popularity.

The prediction is calculated as:

```text
predicted rating = (user average rating + movie average rating) / 2
```

This method is stronger than using only user average or only movie average because it uses information from both sides of the recommendation problem.

It considers:

```text
how the user usually rates movies
how the movie is usually rated by other users
```

For example, if a user usually gives high ratings and the movie also has a high average rating, the predicted rating will be high. If the user is strict or the movie has a low average rating, the prediction will be lower.

In this project, the Combined User-Movie Average baseline performed very well. This shows that simple user and movie rating patterns are highly informative for this dataset.

---

## 4.2 Deep Learning Model: Neural Collaborative Filtering

The deep learning model used in this project is:

```text
Neural Collaborative Filtering
```

Neural Collaborative Filtering is a deep learning approach for recommender systems. It learns hidden relationships between users and movies using embeddings and neural network layers.

Unlike baseline models, which use only simple averages, Neural Collaborative Filtering learns numerical vector representations of users and movies. These vectors are called embeddings.

---

### User and Movie Embeddings

The model has two embedding layers:

```text
User Embedding Layer
Movie Embedding Layer
```

The User Embedding Layer converts each `user_index` into a dense vector. This vector represents hidden user preferences.

For example, the model may learn that one user prefers:

```text
action movies
science fiction movies
high-rated popular movies
```

The Movie Embedding Layer converts each `movie_index` into a dense vector. This vector represents hidden movie characteristics.

For example, the model may learn that one movie is related to:

```text
adventure
comedy
popular family movies
```

These features are not manually written. The model learns them automatically during training.

---

### Neural Network Structure

The model architecture is:

```text
user_index input
      ↓
User Embedding
      ↓
Concatenation
      ↓
Fully Connected Neural Network
      ↓
Predicted Rating
      ↑
Movie Embedding
      ↑
movie_index input
```

First, the model receives two inputs:

```text
user_index
movie_index
```

Then, each input is passed through its embedding layer. The user embedding and movie embedding are combined using concatenation.

The combined vector is passed through fully connected layers. These layers learn non-linear relationships between users and movies.

The final output of the model is:

```text
predicted rating
```

This predicted rating is a numerical value that represents how much the user may like the movie.

---

### Fully Connected Layers

After concatenation, the model uses fully connected neural network layers.

The fully connected layers help the model learn complex patterns such as:

```text
which users have similar preferences
which movies are similar in rating behavior
how user preferences match movie characteristics
```

This is important because recommendation is not always linear. A user may like some movies from one genre but dislike others from the same genre. A neural network can learn more flexible patterns than simple averages.

---

### Dropout Layers

The model also includes dropout layers.

Dropout is a regularization technique. It randomly disables some neurons during training. This helps reduce overfitting.

Overfitting happens when the model learns the training data too well but does not generalize well to new data.

In this project, dropout was used to make the model more stable and improve generalization.

---

### Loss Function

The deep learning model was trained using:

```text
Mean Squared Error Loss
```

MSE Loss is suitable because the task is rating prediction. The model predicts a numerical rating, so the loss measures the difference between the real rating and the predicted rating.

Example:

```text
Real rating: 4.5
Predicted rating: 4.0
Error: 0.5
```

The model tries to minimize this error during training.

---

### Optimizer

The optimizer used was:

```text
Adam
```

Adam was selected because it is commonly used in deep learning and works well for neural network training. It updates model weights based on the calculated loss and helps the model learn faster and more efficiently.

---

### Why Neural Collaborative Filtering Is Appropriate

Neural Collaborative Filtering is appropriate for this project because the dataset contains user-item interactions:

```text
userId → movieId → rating
```

The model learns from these interactions and tries to predict ratings for unseen user-movie pairs.

This method is connected to the main goal of recommendation systems: learning user preferences and ranking items based on predicted interest.

---

## 4.3 Comparison Between Baseline and Deep Learning Model

The baseline models and the deep learning model solve the same problem, but they do it in different ways.

| Method | How It Works | Strength |
|---|---|---|
| Global Average | Predicts one average rating for all pairs | Very simple starting point |
| Movie Average | Uses average rating of each movie | Captures movie popularity |
| User Average | Uses average rating behavior of each user | Captures user rating style |
| Combined Average | Combines user and movie averages | Uses both user and movie information |
| Neural Collaborative Filtering | Learns user and movie embeddings | Can learn complex hidden patterns |

The baseline models are simple and easy to interpret. The deep learning model is more flexible and can learn hidden patterns, but it requires more training and tuning.

In the current experiment, the Combined User-Movie Average baseline achieved the best RMSE and MAE. This means that average-based user and movie patterns were very strong in this dataset.

The Neural Collaborative Filtering model still worked correctly and learned from the data. However, it may need more epochs, a larger training sample, better hyperparameter tuning, or additional features to outperform the strongest baseline.

This comparison is useful because it shows that a deep learning model should not only be implemented, but also evaluated fairly against simple methods.

## 5. Training Setup

The deep learning model was implemented using PyTorch.

The main training settings were:

| Parameter | Value |
|---|---|
| Model | Neural Collaborative Filtering |
| Embedding dimension | 64 |
| Batch size | 4096 |
| Epochs | 5 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Mean Squared Error Loss |
| Dropout | 0.2 |

The model was trained using **Mean Squared Error Loss** because the task is rating prediction. The goal is to minimize the difference between the real rating and the predicted rating.

Adam optimizer was used because it is efficient for neural network training and adapts the learning rate during optimization.

Validation RMSE and MAE were monitored during training. This helped check whether the model was improving and whether overfitting was happening.

The model predictions were limited to the valid MovieLens rating range:

```text
0.5 to 5.0
```

This is important because ratings outside this range are not meaningful for the dataset.

---

## 6. Evaluation Metrics

The project used two evaluation metrics:

```text
RMSE
MAE
```

These metrics are suitable because the project predicts numerical ratings.

### RMSE

RMSE means Root Mean Squared Error. It measures the average prediction error and gives larger penalty to bigger mistakes.

```text
Lower RMSE means better model performance.
```

### MAE

MAE means Mean Absolute Error. It measures the average absolute difference between real ratings and predicted ratings.

```text
Lower MAE means better model performance.
```

Classification metrics such as accuracy, precision, recall, F1-score, and confusion matrix were not used as main metrics because this project does not predict class labels. It predicts continuous rating values.

---

## 7. Results Table

The final comparison of baseline and deep learning models is shown below.

| Model | RMSE | MAE |
|---|---:|---:|
| Global Average Baseline | 1.0522 | 0.8416 |
| Movie Average Baseline | 0.9580 | 0.7436 |
| User Average Baseline | 1.0870 | 0.8341 |
| Combined User-Movie Average Baseline | 0.9482 | 0.7399 |
| Neural Collaborative Filtering | 0.9856 | 0.7757 |

The best result in this experiment was achieved by the **Combined User-Movie Average Baseline**. It had the lowest RMSE and MAE.

The Neural Collaborative Filtering model successfully learned user-movie interaction patterns, but it did not outperform the strongest baseline in the current experiment. This result shows that simple average-based methods can be strong for rating prediction, especially when user and movie averages are informative.

However, the deep learning model still provides value because it introduces a scalable architecture that can be improved with more training, better hyperparameter tuning, larger data samples, and additional features such as genres.

---

## 8. Error Analysis

Error analysis was used to understand where the model performs well and where it makes larger mistakes.

The main error measure was:

```text
absolute_error = |real rating - predicted rating|
```

Small absolute error means that the predicted rating is close to the real rating. Large absolute error means that the model prediction is far from the real user rating.

The model usually performs better when there is enough historical rating information about a user or a movie. It becomes more difficult when a user has unusual rating behavior or when a movie has very few ratings.

Large prediction errors can happen because recommendation data is sparse. Each user rates only a small part of all available movies. Because of this, the model does not always have enough information to understand every user-movie relationship.

The error analysis showed that prediction quality depends on both user history and movie popularity. This is why the combined baseline performed strongly: it uses both user average behavior and movie average rating.

---

## 9. Recommendation Example

The project also includes a simple recommendation example. For a selected user, the system predicts ratings for candidate movies and ranks them by predicted rating.

The recommendation logic is based on the following idea:

```text
higher predicted rating = stronger recommendation
```

This demonstrates how rating prediction can be converted into a ranking task. The model does not only predict ratings; it can also be used to generate a Top-N movie recommendation list.

This is important because the final practical goal of the project is not only to estimate ratings, but also to recommend movies that are most likely to be relevant for a user.

---

## 10. Limitations

This project has several limitations.

First, the model mainly uses user IDs, movie IDs, and rating history. It does not fully use movie descriptions, actors, directors, reviews, or detailed user information.

Second, the deep learning model was trained with limited experimental settings. The number of epochs was small, and the architecture was simple. More tuning could improve the neural model.

Third, MovieLens 20M is a large dataset, so experiments may use sampling to reduce training time. Using the full dataset could improve model performance but requires more computational resources.

Another limitation is the cold-start problem. The model may not recommend well for new users or new movies that have no rating history.

Finally, the user-movie matrix is sparse. Most users rate only a small fraction of all movies, which makes recommendation more difficult.

---

## 11. Conclusion

This project developed a movie rating prediction and recommendation system using the MovieLens 20M dataset.

The project used both simple baseline methods and a deep learning method. The baseline models used average ratings, while the deep learning model used Neural Collaborative Filtering with user and movie embeddings.

The main evaluation metrics were RMSE and MAE. In the current experiment, the Combined User-Movie Average Baseline achieved the best result. The Neural Collaborative Filtering model worked correctly and learned user-movie patterns, but it needs further tuning to outperform the strongest baseline.

Overall, the project demonstrates the full recommendation system workflow:

```text
data preprocessing
baseline modeling
deep learning modeling
training
evaluation
error analysis
recommendation ranking
```

The project can be improved in the future by using more epochs, testing different embedding dimensions, adding movie genres as input features, using a larger dataset sample, and trying more advanced recommender system architectures.

---

## 12. References

1. MovieLens 20M Dataset, GroupLens  
   https://grouplens.org/datasets/movielens/20m/

2. MovieLens 20M README  
   https://files.grouplens.org/datasets/movielens/ml-20m-README.html

3. PyTorch Documentation  
   https://pytorch.org/docs/stable/index.html

4. Scikit-learn Documentation  
   https://scikit-learn.org/stable/
