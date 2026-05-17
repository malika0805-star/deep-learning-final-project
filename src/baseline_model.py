# Baseline model script

# This file will be completed in Week 2.
# Planned baseline methods:
# 1. Global average rating
# 2. Movie average rating
# 3. User average rating

import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.append("src")

from evaluation import evaluate_model, save_results


TRAIN_PATH = "data/processed/train.csv"
VALID_PATH = "data/processed/valid.csv"
TEST_PATH = "data/processed/test.csv"
MOVIES_PATH = "data/movies.csv"

RESULTS_DIR = "results"
MODELS_DIR = "models"


def load_processed_data():
    """
    Load processed train, validation, and test datasets.
    """
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(
            "Processed train.csv not found. Please run src/preprocessing.py first."
        )

    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)
    movies = pd.read_csv(MOVIES_PATH)

    return train_df, valid_df, test_df, movies


def global_average_baseline(train_df, test_df):
    """
    Baseline 1:
    Predict the global average rating for all user-movie pairs.
    """
    global_mean = train_df["rating"].mean()

    predictions = np.full(len(test_df), global_mean)

    result = evaluate_model(
        test_df["rating"],
        predictions,
        "Global Average Baseline"
    )

    return result, global_mean


def movie_average_baseline(train_df, test_df, global_mean):
    """
    Baseline 2:
    Predict rating using the average rating of each movie.
    If a movie is not found in train set, use global average.
    """
    movie_mean_ratings = train_df.groupby("movieId")["rating"].mean()

    predictions = test_df["movieId"].map(movie_mean_ratings)
    predictions = predictions.fillna(global_mean)

    result = evaluate_model(
        test_df["rating"],
        predictions,
        "Movie Average Baseline"
    )

    return result, movie_mean_ratings


def user_average_baseline(train_df, test_df, global_mean):
    """
    Baseline 3:
    Predict rating using the average rating behavior of each user.
    If a user is not found in train set, use global average.
    """
    user_mean_ratings = train_df.groupby("userId")["rating"].mean()

    predictions = test_df["userId"].map(user_mean_ratings)
    predictions = predictions.fillna(global_mean)

    result = evaluate_model(
        test_df["rating"],
        predictions,
        "User Average Baseline"
    )

    return result, user_mean_ratings


def combined_average_baseline(
    test_df,
    global_mean,
    user_mean_ratings,
    movie_mean_ratings
):
    """
    Baseline 4:
    Combine user average and movie average.

    predicted rating = (user average + movie average) / 2
    """
    user_part = test_df["userId"].map(user_mean_ratings).fillna(global_mean)
    movie_part = test_df["movieId"].map(movie_mean_ratings).fillna(global_mean)

    predictions = (user_part + movie_part) / 2

    result = evaluate_model(
        test_df["rating"],
        predictions,
        "Combined User-Movie Average Baseline"
    )

    return result, predictions


def create_example_recommendations(
    train_df,
    test_df,
    movies,
    global_mean,
    user_mean_ratings,
    movie_mean_ratings
):
    """
    Create simple top-10 recommendations for one user.
    """
    example_user_id = test_df["userId"].iloc[0]

    rated_movies = train_df[
        train_df["userId"] == example_user_id
    ]["movieId"].unique()

    candidate_movies = movies[
        ~movies["movieId"].isin(rated_movies)
    ].copy()

    candidate_movies = candidate_movies.sample(
        n=min(1000, len(candidate_movies)),
        random_state=42
    )

    candidate_movies["user_mean"] = user_mean_ratings.get(
        example_user_id,
        global_mean
    )

    candidate_movies["movie_mean"] = candidate_movies["movieId"].map(
        movie_mean_ratings
    ).fillna(global_mean)

    candidate_movies["predicted_rating"] = (
        candidate_movies["user_mean"] + candidate_movies["movie_mean"]
    ) / 2

    recommendations = candidate_movies.sort_values(
        by="predicted_rating",
        ascending=False
    ).head(10)

    return recommendations[["movieId", "title", "genres", "predicted_rating"]]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading processed data...")
    train_df, valid_df, test_df, movies = load_processed_data()

    print("Train shape:", train_df.shape)
    print("Validation shape:", valid_df.shape)
    print("Test shape:", test_df.shape)

    print("\nRunning baseline models...\n")

    global_result, global_mean = global_average_baseline(train_df, test_df)

    movie_result, movie_mean_ratings = movie_average_baseline(
        train_df,
        test_df,
        global_mean
    )

    user_result, user_mean_ratings = user_average_baseline(
        train_df,
        test_df,
        global_mean
    )

    combined_result, combined_predictions = combined_average_baseline(
        test_df,
        global_mean,
        user_mean_ratings,
        movie_mean_ratings
    )

    results = [
        global_result,
        movie_result,
        user_result,
        combined_result
    ]

    results_df = pd.DataFrame(results)

    print("\nBaseline Model Comparison:")
    print(results_df)

    results_csv_path = os.path.join(
        RESULTS_DIR,
        "week2_baseline_results.csv"
    )

    results_txt_path = os.path.join(
        RESULTS_DIR,
        "week2_baseline_results.txt"
    )

    results_df.to_csv(results_csv_path, index=False)
    save_results(results_df, results_txt_path)

    baseline_objects = {
        "global_mean": global_mean,
        "movie_mean_ratings": movie_mean_ratings,
        "user_mean_ratings": user_mean_ratings
    }

    model_path = os.path.join(
        MODELS_DIR,
        "week2_baseline_objects.pkl"
    )

    joblib.dump(baseline_objects, model_path)

    print("\nBaseline objects saved to:", model_path)

    print("\nCreating example recommendations...")

    recommendations = create_example_recommendations(
        train_df,
        test_df,
        movies,
        global_mean,
        user_mean_ratings,
        movie_mean_ratings
    )

    recommendations_path = os.path.join(
        RESULTS_DIR,
        "week2_example_recommendations.csv"
    )

    recommendations.to_csv(recommendations_path, index=False)

    print("Example recommendations saved to:", recommendations_path)

    print("\nWeek 2 baseline modeling completed successfully.")


if __name__ == "__main__":
    main()
