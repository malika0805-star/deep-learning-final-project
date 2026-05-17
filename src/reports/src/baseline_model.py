import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from evaluation import (
    evaluate_classification,
    print_classification_metrics,
    save_confusion_matrix,
    save_metrics_to_txt
)


TRAIN_PATH = "data/processed/train.csv"
VALID_PATH = "data/processed/valid.csv"
TEST_PATH = "data/processed/test.csv"

RESULTS_DIR = "results"
MODEL_DIR = "models"

RANDOM_STATE = 42


def load_processed_data():
    """
    Load train, validation, and test datasets.
    """
    if not os.path.exists(TRAIN_PATH):
        print("Processed data not found.")
        print("Please run preprocessing first:")
        print("python src/preprocessing.py")
        return None, None, None

    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)

    return train_df, valid_df, test_df


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_df, valid_df, test_df = load_processed_data()

    if train_df is None:
        return

    print("Train shape:", train_df.shape)
    print("Validation shape:", valid_df.shape)
    print("Test shape:", test_df.shape)

    X_train = train_df["clean_text"].astype(str)
    y_train = train_df["Like"]

    X_valid = valid_df["clean_text"].astype(str)
    y_valid = valid_df["Like"]

    X_test = test_df["clean_text"].astype(str)
    y_test = test_df["Like"]

    print("\nCreating TF-IDF features...")

    vectorizer = TfidfVectorizer(
        max_features=20000,
        min_df=2,
        max_df=0.90,
        ngram_range=(1, 2)
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_valid_tfidf = vectorizer.transform(X_valid)
    X_test_tfidf = vectorizer.transform(X_test)

    print("TF-IDF train shape:", X_train_tfidf.shape)

    print("\nTraining Logistic Regression baseline model...")

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    model.fit(X_train_tfidf, y_train)

    print("\nEvaluating on validation set...")
    valid_pred = model.predict(X_valid_tfidf)
    valid_metrics = evaluate_classification(y_valid, valid_pred)

    print_classification_metrics(
        valid_metrics,
        title="Validation Results - Logistic Regression Baseline"
    )

    print("\nEvaluating on test set...")
    test_pred = model.predict(X_test_tfidf)
    test_metrics = evaluate_classification(y_test, test_pred)

    print_classification_metrics(
        test_metrics,
        title="Test Results - Logistic Regression Baseline"
    )

    # Save metrics and confusion matrix
    save_metrics_to_txt(
        test_metrics,
        output_path=os.path.join(RESULTS_DIR, "baseline_test_metrics.txt"),
        title="Test Results - Logistic Regression Baseline"
    )

    save_confusion_matrix(
        test_metrics["confusion_matrix"],
        output_path=os.path.join(RESULTS_DIR, "baseline_confusion_matrix.png")
    )

    # Save model and vectorizer
    joblib.dump(model, os.path.join(MODEL_DIR, "baseline_logistic_regression.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

    print("\nBaseline model training completed.")
    print("Saved files:")
    print("- results/baseline_test_metrics.txt")
    print("- results/baseline_confusion_matrix.png")
    print("- models/baseline_logistic_regression.pkl")
    print("- models/tfidf_vectorizer.pkl")


if __name__ == "__main__":
    main()
