import os
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "data/Reviews.csv"
RESULTS_DIR = "results"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print("Dataset file not found.")
        print("Please download Reviews.csv from Kaggle and place it in the data/ folder.")
        print("Expected path:", DATA_PATH)
        return

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("\nDataset loaded successfully.")
    print("Dataset shape:", df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nDataset information:")
    print(df.info())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nRating distribution:")
    print(df["Score"].value_counts().sort_index())

    print("\nBasic statistics for Score:")
    print(df["Score"].describe())

    # Create binary label
    df["Like"] = df["Score"].apply(lambda x: 1 if x >= 4 else 0)

    print("\nBinary label distribution:")
    print(df["Like"].value_counts())

    # Save rating distribution plot
    plt.figure(figsize=(7, 5))
    df["Score"].value_counts().sort_index().plot(kind="bar")
    plt.title("Rating Distribution")
    plt.xlabel("Rating Score")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "rating_distribution.png"))
    plt.close()

    # Save binary label distribution plot
    plt.figure(figsize=(6, 5))
    df["Like"].value_counts().sort_index().plot(kind="bar")
    plt.title("Like / Not Like Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Reviews")
    plt.xticks([0, 1], ["Not Like", "Like"], rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "like_distribution.png"))
    plt.close()

    # Save short dataset summary
    summary_path = os.path.join(RESULTS_DIR, "week1_dataset_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Amazon Fine Food Reviews - Week 1 Dataset Summary\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Dataset shape: {df.shape}\n\n")
        f.write("Columns:\n")
        for col in df.columns:
            f.write(f"- {col}\n")

        f.write("\nMissing values:\n")
        f.write(str(df.isnull().sum()))
        f.write("\n\nRating distribution:\n")
        f.write(str(df["Score"].value_counts().sort_index()))
        f.write("\n\nBinary label distribution:\n")
        f.write(str(df["Like"].value_counts()))

    print("\nWeek 1 exploration completed.")
    print("Saved files:")
    print("- results/rating_distribution.png")
    print("- results/like_distribution.png")
    print("- results/week1_dataset_summary.txt")


if __name__ == "__main__":
    main()
