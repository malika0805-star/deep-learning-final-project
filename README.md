# Movie Rating Prediction and Recommendation Using Neural Collaborative Filtering

---

## Title Page

**Project Title:** Movie Rating Prediction and Recommendation Using Neural Collaborative Filtering  

**Course:** Applied Deep Learning  

**Task Type:** Recommendation / Ranking  

**Dataset:** MovieLens 20M  

**Student Name:** Shpayeva Malika 

**Instructor:** Ardak Shalkarbay-uly 

**University:** Narxoz University  

**Date:** 2026 may  

---

## 1. Project Overview

This project focuses on building a movie recommendation system using the MovieLens 20M dataset. The main task is recommendation and ranking. The model will predict how a user may rate a movie, and these predicted ratings will be used to rank movies and recommend the most relevant movies to each user.

This project is not based on positive or negative sentiment classification. Instead, it uses user-movie rating interactions to learn user preferences and movie patterns.

---

## Project Title

**Movie Rating Prediction and Recommendation Using Neural Collaborative Filtering**

This title clearly describes the main goal of the project. The project is focused on predicting movie ratings and using these predictions to recommend movies to users.

---

## 2. Problem Statement

Online movie platforms contain thousands of movies, so users may spend a lot of time trying to find movies that match their interests. Recommendation systems help solve this problem by predicting what each user may like and ranking items based on predicted user preference.

The main problem in this project is to predict the rating that a specific user may give to a specific movie. After predicting ratings for many movies, the system can rank movies and recommend the movies with the highest predicted ratings.

This project is useful because recommendation systems are widely used in streaming platforms, online stores, music services, and social media. A good recommendation model can improve user experience by helping users find relevant content faster.

### Input

```text
userId
movieId
```

### Output

```text
predicted rating
```

### Prediction Target

```text
rating
```

The target value is a numerical rating from 0.5 to 5.0. This means the project is not a positive/negative classification task. It is a recommendation/ranking task based on rating prediction.

### Example

```text
Input:
userId = 15
movieId = 318

Output:
predicted rating = 4.7
```

The predicted ratings will be used to rank movies and recommend the top movies to each user.

---

## 3. Dataset

### Dataset Name

```text
MovieLens 20M
```

### Dataset Source

Official dataset page:

```text
https://grouplens.org/datasets/movielens/20m/
```

Direct download link:

```text
https://files.grouplens.org/datasets/movielens/ml-20m.zip
```

### Dataset Description

MovieLens 20M is a real and large dataset for recommendation system tasks. It contains user ratings for movies and is suitable for training and evaluating recommendation models.

The dataset includes:

```text
20,000,263 ratings
138,493 users
27,278 movies
```

The rating scale is:

```text
0.5 to 5.0
```

This dataset is relevant to the project because it contains user-item interactions:

```text
userId → movieId → rating
```

These interactions allow the model to learn user preferences and movie patterns.

### Main Files Used

```text
ratings.csv
movies.csv
```

### ratings.csv

| Column | Description |
|---|---|
| userId | Unique user identifier |
| movieId | Unique movie identifier |
| rating | Rating value from 0.5 to 5.0 |
| timestamp | Time when the rating was given |

### movies.csv

| Column | Description |
|---|---|
| movieId | Unique movie identifier |
| title | Movie title |
| genres | Movie genres |

### Input Features

```text
userId
movieId
```

Additional movie information such as `title` and `genres` may be used for analysis and explanation.

### Target Output

```text
rating
```

### Data Format

```text
CSV
```

### License / Usage Notes

The dataset is publicly available from GroupLens for research and educational use. Large raw dataset files will not be committed to GitHub. Instead, the repository will include download instructions in the `data/README.md` file.

---

## 4. Planned Method

The project will include a simple baseline model and one deep learning model. The baseline model will be used as a comparison point, and the deep learning model will be used to improve rating prediction.

---

### 4.1 Baseline Model

The baseline model will use average rating methods.

Planned baseline methods:

```text
Global average rating
Movie average rating
User average rating
Combined user-movie average rating
```

The simplest baseline predicts the same average rating for all user-movie pairs. A stronger baseline predicts ratings based on the average rating of each movie or the average rating behavior of each user.

This baseline is useful because it gives a simple result that the deep learning model should improve.

---

### 4.2 Deep Learning Model

The main deep learning model will be:

```text
Neural Collaborative Filtering
```

This model is appropriate because the project is based on user-item interactions. Neural Collaborative Filtering can learn hidden relationships between users and movies using embedding layers.

### Model Architecture

```text
userId input
      ↓
User Embedding
      ↓
Concatenation
      ↓
Fully Connected Layers
      ↓
Predicted Rating
      ↑
Movie Embedding
      ↑
movieId input
```

### Main Components

```text
User Embedding Layer
Movie Embedding Layer
Concatenation Layer
Fully Connected Neural Network
Output Layer
```

The user embedding will learn hidden user preferences.  
The movie embedding will learn hidden movie features.  
The fully connected layers will combine these embeddings and predict the rating.

---

### 4.3 Loss Function

The loss function will be:

```text
Mean Squared Error Loss
```

This loss function is suitable because the model predicts numerical rating values.

The goal is to reduce the difference between the real rating and the predicted rating.

Example:

```text
Real rating: 4.5
Predicted rating: 4.2
Error: 0.3
```

---

### 4.4 Evaluation Metrics

The main evaluation metrics will be:

```text
RMSE
MAE
```

### RMSE

RMSE means Root Mean Squared Error. It measures prediction error and gives stronger penalty to large mistakes.

```text
Lower RMSE = better model performance
```

### MAE

MAE means Mean Absolute Error. It measures the average absolute difference between real ratings and predicted ratings.

```text
Lower MAE = better model performance
```

Accuracy, precision, recall, F1-score, and confusion matrix are not the main metrics for this project because the task is not classification. The model predicts numerical ratings.

---

### 4.5 Train / Validation / Test Split Plan

The dataset will be split into three parts:

| Dataset Part | Percentage | Purpose |
|---|---:|---|
| Training set | 70% | Used to train the model |
| Validation set | 15% | Used to tune model parameters |
| Test set | 15% | Used for final evaluation |

The test set will not be used during training or model selection. It will only be used at the final evaluation stage.

---

## 5. Expected Challenges

One possible challenge is the large size of the MovieLens 20M dataset. It may require more memory and longer training time, so a sample of the dataset may be used during early experiments.

Another challenge is sparsity. Each user rates only a small number of movies compared to the total number of movies. This makes recommendation more difficult because many user-movie pairs have no rating.

The project may also face the cold-start problem. The model may not perform well for new users or new movies that do not have rating history.

Overfitting is another possible issue in the deep learning model. To reduce it, validation results will be monitored, and regularization techniques such as dropout may be used if necessary.

---

## 6. Weekly Plan

| Week | Planned Work | Expected Output |
|---|---|---|
| Week 1 | Dataset selection, GitHub repository setup, project structure, dataset instructions, exploratory data analysis | README, data/README.md, Week 1 report, dataset summary |
| Week 2 | Data preprocessing, userId and movieId encoding, train/validation/test split, baseline model | Processed data, baseline RMSE/MAE results, Week 2 report |
| Week 3 | Neural Collaborative Filtering model implementation, training loop, experiments with embeddings and hidden layers | Deep learning model results, training plots, Week 3 report |
| Week 4 | Final evaluation, baseline vs deep learning comparison, error analysis, final report and presentation/demo | Final code, model comparison, final report, presentation/demo |

---

## 7. Repository Plan

The project repository will include:

```text
source code
notebooks for experiments
dataset loading instructions
model training code
evaluation code
weekly progress reports
final report
README with running instructions
```

Recommended repository structure:

```text
movie-rating-recommendation/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   ├── week1_dataset_exploration.ipynb
│   ├── week2_preprocessing_baseline.ipynb
│   └── week3_deep_learning_model.ipynb
├── src/
│   ├── explore_data.py
│   ├── preprocessing.py
│   ├── baseline_model.py
│   ├── deep_learning_model.py
│   └── evaluation.py
├── reports/
│   ├── week-01.md
│   ├── week-02.md
│   ├── week-03.md
│   └── week-04.md
├── results/
│   └── .gitkeep
├── models/
│   └── .gitkeep
└── final-report.md
```

Large dataset files such as `ratings.csv`, `movies.csv`, and `ml-20m.zip` will not be committed to GitHub. They will be stored locally, and download instructions will be provided in the repository.

---

## 8. References

1. MovieLens 20M Dataset, GroupLens  
   https://grouplens.org/datasets/movielens/20m/

2. MovieLens 20M README  
   https://files.grouplens.org/datasets/movielens/ml-20m-README.html
