# Data Directory

This folder is used for storing dataset files for the project.

The project uses the **MovieLens 20M** dataset from GroupLens. The dataset is used for a **recommendation / ranking** task, where the model predicts movie ratings for user-movie pairs.

---

## Dataset Name

```text
MovieLens 20M
```

---

## Dataset Source

Official dataset page:

```text
https://grouplens.org/datasets/movielens/20m/
```

Direct download link:

```text
https://files.grouplens.org/datasets/movielens/ml-20m.zip
```

---

## Dataset Information

MovieLens 20M is a large movie rating dataset commonly used for recommender system research and experiments.

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

This dataset is suitable for this project because it contains user-item interactions:

```text
userId → movieId → rating
```

These interactions allow the model to learn user preferences and predict ratings for unseen user-movie pairs.

---

## Files Used in This Project

After downloading and extracting the dataset, this project uses the following files:

```text
ratings.csv
movies.csv
```

---

## File Descriptions

### ratings.csv

This is the main file for model training.

It contains:

| Column | Description |
|---|---|
| userId | Unique user identifier |
| movieId | Unique movie identifier |
| rating | Rating value from 0.5 to 5.0 |
| timestamp | Time when the rating was given |

Example structure:

```text
userId,movieId,rating,timestamp
1,2,3.5,1112486027
1,29,3.5,1112484676
```

---

### movies.csv

This file contains movie information.

It contains:

| Column | Description |
|---|---|
| movieId | Unique movie identifier |
| title | Movie title |
| genres | Movie genres |

Example structure:

```text
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
```

---

## Download Instructions

### Option 1: Download from Official Website

1. Open the official MovieLens 20M dataset page:

```text
https://grouplens.org/datasets/movielens/20m/
```

2. Download the dataset archive:

```text
ml-20m.zip
```

3. Extract the ZIP file.

4. Copy the following files into this `data/` folder:

```text
ratings.csv
movies.csv
```

---

### Option 2: Direct Download Link

Open this link in your browser:

```text
https://files.grouplens.org/datasets/movielens/ml-20m.zip
```

Then extract the ZIP file and copy:

```text
ratings.csv
movies.csv
```

into the `data/` folder.

---

### Option 3: Download Using Terminal

From the project root folder, run:

```bash
curl -L -o data/ml-20m.zip https://files.grouplens.org/datasets/movielens/ml-20m.zip
```

Then unzip:

```bash
unzip data/ml-20m.zip -d data/
```

After extracting, copy the needed files:

```bash
cp data/ml-20m/ratings.csv data/ratings.csv
cp data/ml-20m/movies.csv data/movies.csv
```

---

## Expected Data Folder Structure

After preparing the dataset, the `data/` folder should look like this:

```text
data/
├── README.md
├── ratings.csv
└── movies.csv
```

The extracted folder and ZIP file are not required after copying the files.

Optional files that can be deleted after extraction:

```text
ml-20m.zip
ml-20m/
```

---

## Important Note About GitHub

Large dataset files should **not** be committed to GitHub.

Do not upload these files:

```text
ratings.csv
movies.csv
ml-20m.zip
ml-20m/
```

These files should stay only on the local computer.

The repository uses `.gitignore` to prevent large dataset files from being committed.

Recommended `.gitignore` rules:

```gitignore
data/*.csv
data/*.zip
data/ml-20m/
```

---

## Why This Dataset Is Suitable

MovieLens 20M is suitable for this project because:

```text
it is a real-world recommendation dataset
it contains many users and movies
it has millions of ratings
it supports rating prediction
it can be used for recommendation and ranking
```

The model will use:

```text
Input: userId, movieId
Output: predicted rating
```

Predicted ratings can later be used to rank movies and recommend the top movies to each user.

---

## Project Task Connection

This dataset supports the project task:

```text
Recommendation / Ranking
```

The final model will learn patterns from user-movie rating history and predict how much a user may like a movie.

The main evaluation metrics will be:

```text
RMSE
MAE
```

These metrics are appropriate because the model predicts numerical rating values.
