import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error
)


def evaluate_classification(y_true, y_pred):
    """
    Evaluate classification model using required metrics:
    - accuracy
    - precision
    - recall
    - F1-score
    - confusion matrix
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }

    return metrics


def evaluate_regression(y_true, y_pred):
    """
    Evaluate rating prediction using:
    - RMSE
    - MAE
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    metrics = {
        "rmse": rmse,
        "mae": mae
    }

    return metrics


def print_classification_metrics(metrics, title="Classification Results"):
    print("\n" + title)
    print("=" * len(title))
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])


def save_confusion_matrix(cm, output_path):
    """
    Save confusion matrix as an image.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0, 1], ["Not Like", "Like"])
    plt.yticks([0, 1], ["Not Like", "Like"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_metrics_to_txt(metrics, output_path, title="Model Results"):
    """
    Save classification metrics into a text file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(title + "\n")
        f.write("=" * len(title) + "\n\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-score:  {metrics['f1_score']:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(metrics["confusion_matrix"]))
