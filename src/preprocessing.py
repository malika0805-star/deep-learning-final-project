import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = "data/Reviews.csv"
OUTPUT_DIR = "data/processed"

# For faster training during Week 2, we can use a sample.
# If you want to use the full dataset later, change SAMPLE_SIZE to None.
SAMPLE_SIZE = 100000
RANDOM_STATE = 42


def clean_text(text):
    """
    Clean review text:
    - convert to lowercase
    - remove HTML tags
    - remove non-letter symbols
    - remove extra spaces
    """
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print("Dataset file not found.")
        print("Please download Reviews.csv and place it in the data/ folder.")
        print("Expected path:", DATA_PATH)
        return

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", df.shape)

    # Select useful columns
    selected_columns = [
        "Id",
        "ProductId",
        "UserId",
        "Score",
        "Summary",
        "Text"
    ]

    df = df[selected_columns]

    print("Selected columns:")
    print(df.columns.tolist())

    # Remove missing values
    df = df.dropna(subset=["ProductId", "UserId", "Score", "Text"])

    print("Shape after removing missing values:", df.shape)

    # Use sample for faster experiments
    if SAMPLE_SIZE is not None and len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    print("Shape after sampling:", df.shape)

    # Clean text
    print("Cleaning review text...")
    df["clean_text"] = df["Text"].apply(clean_text)

    # Remove empty cleaned texts
    df = df[df["clean_text"].str.len() > 0]

    # Create binary label
    # Score >= 4 -> Like = 1
    # Score < 4  -> Not Like = 0
    df["Like"] = df["Score"].apply(lambda x: 1 if x >= 4 else 0)

    print("Binary label distribution:")
    print(df["Like"].value_counts())

    # Split features and target
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["Like"]
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["Like"]
    )

    print("Train shape:", train_df.shape)
    print("Validation shape:", valid_df.shape)
    print("Test shape:", test_df.shape)

    # Save processed data
    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    valid_path = os.path.join(OUTPUT_DIR, "valid.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Processed files saved:")
    print(train_path)
    print(valid_path)
    print(test_path)


if __name__ == "__main__":
    main()
