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

## 2. Problem Statement

Online movie platforms contain a very large number of movies. Because of this, users may have difficulty finding movies that match their interests. Recommendation systems help solve this problem by predicting what a user may like and ranking items based on user preferences.

The problem in this project is to predict the rating that a specific user may give to a specific movie. After predicting ratings for many movies, the system can recommend movies with the highest predicted ratings.

### Input

```text
userId
movieId
```

### Output

```text
predicted rating
```

### Example

```text
Input:
userId = 15
movieId = 318

Output:
predicted rating = 4.7
```

The final system will rank movies according to predicted ratings and recommend the top movies to the user.

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

MovieLens 20M is a large and meaningful dataset for recommendation system tasks. It contains millions of movie ratings from users.

The dataset includes:

```text
20,000,263 ratings
138,493 users
27,278 movies
```

The rating values are given on a scale from:

```text
0.5 to 5.0
```

This dataset is suitable for training and evaluation because it is large, real, and directly connected to recommendation and ranking tasks.

---

## 4. Dataset Files

The main files used in this project are:

```text
ratings.csv
movies.csv
```

### ratings.csv

This file contains user-movie rating interactions.

| Column | Description |
|---|---|
| userId | Unique identifier of a user |
| movieId | Unique identifier of a movie |
| rating | Rating value from 0.5 to 5.0 |
| timestamp | Time when the rating was given |

### movies.csv

This file contains information about movies.

| Column | Description |
|---|---|
| movieId | Unique identifier of a movie |
| title | Movie title |
| genres | Movie genres |

---

## 5. Target Variable

The target variable is:

```text
rating
```

The model predicts a numerical rating for a user-movie pair.

This is a rating prediction task that will be used for recommendation and ranking.

---

## 6. Planned Method

The project will include a simple baseline model and a deep learning model.

---

### 6.1 Baseline Model

The baseline model will use simple average rating methods.

Possible baseline methods:

```text
Global average rating
Movie average rating
User average rating
User-movie average combination
```

The baseline model is important because it gives a simple result for comparison. If the deep learning model performs better than the baseline, it means the neural network learned useful patterns from the data.

---

### 6.2 Deep Learning Model

The main deep learning model will be:

```text
Neural Collaborative Filtering
```

The model will use embeddings for users and movies. These embeddings will represent users and movies as numerical vectors.

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

The model will learn hidden relationships between users and movies. For example, if two users have similar rating behavior, the model may learn similar embeddings for them.

---

## 7. Loss Function

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

## 8. Evaluation Metrics

The main evaluation metrics will be:

```text
RMSE
MAE
```

### RMSE

RMSE means Root Mean Squared Error. It measures the average prediction error and gives more penalty to large errors.

```text
Lower RMSE means better model performance.
```

### MAE

MAE means Mean Absolute Error. It measures the average absolute difference between real ratings and predicted ratings.

```text
Lower MAE means better model performance.
```

Other metrics such as accuracy, precision, recall, F1-score, and confusion matrix are not the main metrics for this project because the task is not classification. The project predicts numerical ratings.

---

## 9. Train / Validation / Test Split

The dataset will be split into three parts:

| Dataset Part | Percentage | Purpose |
|---|---:|---|
| Training set | 70% | Used to train the model |
| Validation set | 15% | Used to tune model parameters |
| Test set | 15% | Used for final evaluation |

The test set will not be used during training or model selection. It will only be used for final evaluation.

---

## 10. Repository Structure

The project repository will be organized as follows:

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

---

## 11. Repository Content

The repository will include:

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

Large raw dataset files will not be uploaded to GitHub. Instead, the repository includes dataset download instructions.

---

## 12. Dataset Preparation

Download the MovieLens 20M dataset from:

```text
https://files.grouplens.org/datasets/movielens/ml-20m.zip
```

Extract the ZIP file.

Copy the following files into the `data/` folder:

```text
ratings.csv
movies.csv
```

The final data folder should look like this:

```text
data/
├── README.md
├── ratings.csv
└── movies.csv
```

The dataset files are ignored by `.gitignore`, so they should not be committed to GitHub.

---

## 13. Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/movie-rating-recommendation.git
cd movie-rating-recommendation
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment

For macOS / Linux:

```bash
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## 14. Required Libraries

The project uses the following Python libraries:

```text
pandas
numpy
matplotlib
scikit-learn
torch
tqdm
joblib
```

---

## 15. How to Run the Project

### 1. Explore the Dataset

```bash
python src/explore_data.py
```

This script checks:

```text
dataset shape
column names
missing values
rating distribution
number of unique users
number of unique movies
ratings per user
ratings per movie
```

### 2. Run Data Preprocessing

```bash
python src/preprocessing.py
```

This script prepares the dataset for model training.

Main preprocessing steps:

```text
load ratings.csv
load movies.csv
check missing values
encode userId
encode movieId
split data into train, validation, and test sets
save processed data
```

### 3. Run Baseline Model

```bash
python src/baseline_model.py
```

This script trains and evaluates the baseline model.

### 4. Run Deep Learning Model

```bash
python src/deep_learning_model.py
```

This script trains the Neural Collaborative Filtering model.

### 5. Run Evaluation

```bash
python src/evaluation.py
```

This script calculates RMSE and MAE for the final model.

---

## 16. Expected Output

The final project will produce:

```text
dataset summary
rating distribution plot
baseline model results
deep learning model results
RMSE and MAE values
model comparison table
final report
weekly progress reports
```

Expected result files:

```text
results/week1_rating_distribution.png
results/baseline_metrics.txt
results/deep_learning_metrics.txt
results/model_comparison.csv
models/baseline_model.pkl
models/neural_collaborative_filtering.pt
```

---

## 17. Weekly Plan

| Week | Planned Work | Expected Output |
|---|---|---|
| Week 1 | Dataset selection, repository setup, dataset exploration | README, dataset instructions, Week 1 report, dataset summary |
| Week 2 | Data preprocessing, train/validation/test split, baseline model | Processed data, baseline RMSE/MAE, Week 2 report |
| Week 3 | Neural Collaborative Filtering model training and experiments | Deep learning model results, plots, Week 3 report |
| Week 4 | Final evaluation, comparison, final report and presentation | Final code, final report, model comparison, slides/demo |

---

## 18. GitHub Commit Plan

The project will be developed gradually over one month. Each week will include meaningful commits.

### Good Commit Examples

```text
add dataset loading code
add MovieLens data exploration notebook
add preprocessing script
add baseline model
add neural collaborative filtering model
add evaluation metrics
add model comparison table
update weekly report
improve README instructions
fix data split bug
```

### Weak Commit Examples

```text
update
final
changes
upload files
```

The repository should show regular progress, not one final upload at the end.

---

## 19. Expected Challenges

Possible challenges in this project include:

```text
large dataset size
long model training time
memory limitations
sparse user-movie interactions
cold-start problem for new users or movies
overfitting in the deep learning model
```

MovieLens 20M is large, so the project may use sampling during early experiments. Another challenge is that each user rates only a small number of movies compared to the total number of movies, so the user-movie interaction matrix is sparse.

---

## 20. Limitations

This project mainly uses rating history. It does not fully use additional information such as movie descriptions, actors, directors, or user demographic information.

The model may also have difficulty recommending movies to new users or new movies with no rating history. This is known as the cold-start problem.

---

## 21. Future Improvements

Possible future improvements include:

```text
using movie genres as additional input features
testing different embedding sizes
adding dropout to reduce overfitting
using timestamps for time-aware recommendation
building a Top-N recommendation demo
comparing more recommendation algorithms
```

---

## 22. Final Deliverables

The final submission will include:

```text
GitHub repository
README.md
dataset instructions
source code
experiment notebooks
weekly reports
final report
saved results
trained model files
presentation or demo
```

---

## 23. References

1. MovieLens 20M Dataset, GroupLens  
   https://grouplens.org/datasets/movielens/20m/

2. MovieLens 20M README  
   https://files.grouplens.org/datasets/movielens/ml-20m-README.html
