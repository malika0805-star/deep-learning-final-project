# Data preprocessing script for MovieLens 20M

# This file will be completed in Week 2.
# Main tasks:
# 1. Load ratings.csv
# 2. Encode userId and movieId
# 3. Split data into train, validation, and test sets
# 4. Save processed data

import os
import pandas as pd
from sklearn.model_selection import train_test_split


RATINGS_PATH = "data/ratings.csv"
MOVIES_PATH = "data/movies.csv"

PROCESSED_DIR = "data/processed"

SAMPLE_SIZE = 500_000
RANDOM_STATE = 42


def load_data():
    """
    Load ratings.csv and movies.csv.
    """
    if not os.path.exists(RATINGS_PATH):
        raise FileNotFoundError(
            "ratings.csv not found. Please place it in the data/ folder."
        )

    if not os.path.exists(MOVIES_PATH):
        raise FileNotFoundError(
            "movies.csv not found. Please place it in the data/ folder."
        )

    ratings = pd.read_csv(RATINGS_PATH)
    movies = pd.read_csv(MOVIES_PATH)

    return ratings, movies


def preprocess_data(ratings, movies):
    """
    Select important columns, remove missing values,
    sample data, and encode userId and movieId.
    """
    ratings = ratings[["userId", "movieId", "rating", "timestamp"]].copy()
    movies = movies[["movieId", "title", "genres"]].copy()

    ratings = ratings.dropna()
    movies = movies.dropna()

    if SAMPLE_SIZE is not None and len(ratings) > SAMPLE_SIZE:
        ratings = ratings.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    ratings = ratings.reset_index(drop=True)

    ratings["user_index"] = ratings["userId"].astype("category").cat.codes
    ratings["movie_index"] = ratings["movieId"].astype("category").cat.codes

    return ratings, movies


def save_mappings(ratings):
    """
    Save mapping tables for userId and movieId.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    user_mapping = ratings[["userId", "user_index"]].drop_duplicates()
    movie_mapping = ratings[["movieId", "movie_index"]].drop_duplicates()

    user_mapping.to_csv(
        os.path.join(PROCESSED_DIR, "user_mapping.csv"),
        index=False
    )

    movie_mapping.to_csv(
        os.path.join(PROCESSED_DIR, "movie_mapping.csv"),
        index=False
    )

    print("Mapping files saved.")


def split_data(ratings):
    """
    Split dataset into train, validation, and test sets.

    Train: 70%
    Validation: 15%
    Test: 15%
    """
    train_df, temp_df = train_test_split(
        ratings,
        test_size=0.30,
        random_state=RANDOM_STATE
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE
    )

    return train_df, valid_df, test_df


def save_processed_data(train_df, valid_df, test_df):
    """
    Save processed train, validation, and test datasets.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    valid_path = os.path.join(PROCESSED_DIR, "valid.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Processed datasets saved:")
    print(train_path)
    print(valid_path)
    print(test_path)


def main():
    print("Loading data...")
    ratings, movies = load_data()

    print("Original ratings shape:", ratings.shape)
    print("Original movies shape:", movies.shape)

    print("\nPreprocessing data...")
    ratings, movies = preprocess_data(ratings, movies)

    print("Processed ratings shape:", ratings.shape)
    print("Processed movies shape:", movies.shape)

    print("\nNumber of encoded users:", ratings["user_index"].nunique())
    print("Number of encoded movies:", ratings["movie_index"].nunique())

    print("\nSaving mapping files...")
    save_mappings(ratings)

    print("\nSplitting data...")
    train_df, valid_df, test_df = split_data(ratings)

    print("Train shape:", train_df.shape)
    print("Validation shape:", valid_df.shape)
    print("Test shape:", test_df.shape)

    print("\nSaving processed data...")
    save_processed_data(train_df, valid_df, test_df)

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()
