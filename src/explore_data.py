import os
import pandas as pd
import matplotlib.pyplot as plt


RATINGS_PATH = "data/ratings.csv"
MOVIES_PATH = "data/movies.csv"
RESULTS_DIR = "results"


def check_file_exists(file_path, file_name):
    if not os.path.exists(file_path):
        print(f"{file_name} not found.")
        print(f"Please place {file_name} inside the data folder.")
        print(f"Expected path: {file_path}")
        return False
    return True


def load_data():
    ratings = pd.read_csv(RATINGS_PATH)
    movies = pd.read_csv(MOVIES_PATH)
    return ratings, movies


def explore_basic_info(ratings, movies):
    print("Ratings dataset shape:", ratings.shape)
    print("Movies dataset shape:", movies.shape)

    print("\nFirst 5 rows of ratings:")
    print(ratings.head())

    print("\nFirst 5 rows of movies:")
    print(movies.head())

    print("\nRatings columns:")
    print(ratings.columns.tolist())

    print("\nMovies columns:")
    print(movies.columns.tolist())


def check_missing_values(ratings, movies):
    print("\nMissing values in ratings:")
    print(ratings.isnull().sum())

    print("\nMissing values in movies:")
    print(movies.isnull().sum())


def analyze_ratings(ratings):
    rating_distribution = ratings["rating"].value_counts().sort_index()

    print("\nRating distribution:")
    print(rating_distribution)

    print("\nRating statistics:")
    print(ratings["rating"].describe())

    print("\nNumber of unique users:")
    print(ratings["userId"].nunique())

    print("\nNumber of unique movies in ratings:")
    print(ratings["movieId"].nunique())

    return rating_distribution


def analyze_movies(movies):
    print("\nNumber of movies in movies.csv:")
    print(movies["movieId"].nunique())

    print("\nFirst 10 genre values:")
    print(movies["genres"].head(10))

    genre_counts = {}

    for genre_list in movies["genres"]:
        genres = str(genre_list).split("|")
        for genre in genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

    genre_counts_df = pd.DataFrame(
        list(genre_counts.items()),
        columns=["genre", "count"]
    )

    genre_counts_df = genre_counts_df.sort_values(
        by="count",
        ascending=False
    )

    print("\nGenre distribution:")
    print(genre_counts_df)

    return genre_counts_df


def save_rating_distribution_plot(rating_distribution):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    plt.figure(figsize=(8, 5))
    rating_distribution.plot(kind="bar")
    plt.title("MovieLens 20M Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Number of Ratings")
    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, "week1_rating_distribution.png")
    plt.savefig(output_path)
    plt.close()

    print(f"\nRating distribution plot saved to: {output_path}")


def save_genre_distribution_plot(genre_counts_df):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.bar(genre_counts_df["genre"], genre_counts_df["count"])
    plt.title("Movie Genre Distribution")
    plt.xlabel("Genre")
    plt.ylabel("Number of Movies")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, "week1_genre_distribution.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Genre distribution plot saved to: {output_path}")


def save_dataset_summary(ratings, movies, rating_distribution, genre_counts_df):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary_path = os.path.join(RESULTS_DIR, "week1_dataset_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("MovieLens 20M - Week 1 Dataset Summary\n")
        file.write("=" * 50 + "\n\n")

        file.write(f"Ratings dataset shape: {ratings.shape}\n")
        file.write(f"Movies dataset shape: {movies.shape}\n\n")

        file.write("Ratings columns:\n")
        for column in ratings.columns:
            file.write(f"- {column}\n")

        file.write("\nMovies columns:\n")
        for column in movies.columns:
            file.write(f"- {column}\n")

        file.write("\nMissing values in ratings:\n")
        file.write(str(ratings.isnull().sum()))

        file.write("\n\nMissing values in movies:\n")
        file.write(str(movies.isnull().sum()))

        file.write("\n\nRating distribution:\n")
        file.write(str(rating_distribution))

        file.write("\n\nRating statistics:\n")
        file.write(str(ratings["rating"].describe()))

        file.write("\n\nNumber of unique users:\n")
        file.write(str(ratings["userId"].nunique()))

        file.write("\n\nNumber of unique movies in ratings:\n")
        file.write(str(ratings["movieId"].nunique()))

        file.write("\n\nNumber of movies in movies.csv:\n")
        file.write(str(movies["movieId"].nunique()))

        file.write("\n\nGenre distribution:\n")
        file.write(str(genre_counts_df))

    print(f"Dataset summary saved to: {summary_path}")


def main():
    print("Week 1 Dataset Exploration")
    print("=" * 30)

    if not check_file_exists(RATINGS_PATH, "ratings.csv"):
        return

    if not check_file_exists(MOVIES_PATH, "movies.csv"):
        return

    ratings, movies = load_data()

    explore_basic_info(ratings, movies)
    check_missing_values(ratings, movies)

    rating_distribution = analyze_ratings(ratings)
    genre_counts_df = analyze_movies(movies)

    save_rating_distribution_plot(rating_distribution)
    save_genre_distribution_plot(genre_counts_df)

    save_dataset_summary(
        ratings,
        movies,
        rating_distribution,
        genre_counts_df
    )

    print("\nWeek 1 dataset exploration completed successfully.")


if __name__ == "__main__":
    main()
