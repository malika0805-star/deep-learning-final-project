import os
import pandas as pd
import matplotlib.pyplot as plt


RATINGS_PATH = "data/ratings.csv"
MOVIES_PATH = "data/movies.csv"
RESULTS_DIR = "results"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(RATINGS_PATH):
        print("ratings.csv not found.")
        print("Please place ratings.csv inside the data folder.")
        return

    if not os.path.exists(MOVIES_PATH):
        print("movies.csv not found.")
        print("Please place movies.csv inside the data folder.")
        return

    ratings = pd.read_csv(RATINGS_PATH)
    movies = pd.read_csv(MOVIES_PATH)

    print("Ratings shape:", ratings.shape)
    print("Movies shape:", movies.shape)

    print("\nRatings columns:")
    print(ratings.columns.tolist())

    print("\nMovies columns:")
    print(movies.columns.tolist())

    print("\nMissing values in ratings:")
    print(ratings.isnull().sum())

    print("\nMissing values in movies:")
    print(movies.isnull().sum())

    rating_distribution = ratings["rating"].value_counts().sort_index()

    print("\nRating distribution:")
    print(rating_distribution)

    print("\nNumber of unique users:")
    print(ratings["userId"].nunique())

    print("\nNumber of unique movies:")
    print(ratings["movieId"].nunique())

    plt.figure(figsize=(8, 5))
    rating_distribution.plot(kind="bar")
    plt.title("MovieLens 20M Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Number of Ratings")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "week1_rating_distribution.png"))
    plt.close()

    summary_path = os.path.join(RESULTS_DIR, "week1_dataset_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("MovieLens 20M Dataset Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Ratings shape: {ratings.shape}\n")
        f.write(f"Movies shape: {movies.shape}\n")
        f.write(f"Unique users: {ratings['userId'].nunique()}\n")
        f.write(f"Unique movies: {ratings['movieId'].nunique()}\n\n")
        f.write("Rating distribution:\n")
        f.write(str(rating_distribution))

    print("\nDataset exploration completed.")
    print("Saved files:")
    print("results/week1_rating_distribution.png")
    print("results/week1_dataset_summary.txt")


if __name__ == "__main__":
    main()
