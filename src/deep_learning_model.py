# Deep learning model script

# This file will be completed in Week 3.
# Planned model:
# Neural Collaborative Filtering

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error


TRAIN_PATH = "data/processed/train.csv"
VALID_PATH = "data/processed/valid.csv"
TEST_PATH = "data/processed/test.csv"

RESULTS_DIR = "results"
MODELS_DIR = "models"

BATCH_SIZE = 4096
EPOCHS = 5
EMBEDDING_DIM = 64
LEARNING_RATE = 0.001


class MovieRatingDataset(Dataset):
    """
    PyTorch Dataset for movie rating prediction.

    Each sample contains:
    - user_index
    - movie_index
    - rating
    """

    def __init__(self, dataframe):
        self.users = torch.tensor(
            dataframe["user_index"].values,
            dtype=torch.long
        )

        self.movies = torch.tensor(
            dataframe["movie_index"].values,
            dtype=torch.long
        )

        self.ratings = torch.tensor(
            dataframe["rating"].values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, index):
        return self.users[index], self.movies[index], self.ratings[index]


class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering model.

    The model uses:
    - user embedding
    - movie embedding
    - fully connected neural network
    - predicted rating output
    """

    def __init__(self, num_users, num_movies, embedding_dim=64):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, user_ids, movie_ids):
        user_vector = self.user_embedding(user_ids)
        movie_vector = self.movie_embedding(movie_ids)

        combined_vector = torch.cat(
            [user_vector, movie_vector],
            dim=1
        )

        predicted_rating = self.network(combined_vector)

        return predicted_rating.squeeze()


def get_device():
    """
    Select available device for training.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_processed_data():
    """
    Load train, validation, and test datasets.
    """

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(
            "train.csv not found. Please run src/preprocessing.py first."
        )

    if not os.path.exists(VALID_PATH):
        raise FileNotFoundError(
            "valid.csv not found. Please run src/preprocessing.py first."
        )

    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            "test.csv not found. Please run src/preprocessing.py first."
        )

    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)

    return train_df, valid_df, test_df


def get_number_of_users_and_movies(train_df, valid_df, test_df):
    """
    Get number of users and movies for embedding layers.
    """

    num_users = int(
        max(
            train_df["user_index"].max(),
            valid_df["user_index"].max(),
            test_df["user_index"].max()
        ) + 1
    )

    num_movies = int(
        max(
            train_df["movie_index"].max(),
            valid_df["movie_index"].max(),
            test_df["movie_index"].max()
        ) + 1
    )

    return num_users, num_movies


def create_dataloaders(train_df, valid_df, test_df):
    """
    Create PyTorch DataLoader objects.
    """

    train_dataset = MovieRatingDataset(train_df)
    valid_dataset = MovieRatingDataset(valid_df)
    test_dataset = MovieRatingDataset(test_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, valid_loader, test_loader


def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error.
    """
    return mean_absolute_error(y_true, y_pred)


def evaluate_model(model, data_loader, device):
    """
    Evaluate model using RMSE and MAE.
    """

    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for users, movies, ratings in data_loader:
            users = users.to(device)
            movies = movies.to(device)
            ratings = ratings.to(device)

            predictions = model(users, movies)

            predictions = predictions.detach().cpu().numpy()
            ratings = ratings.detach().cpu().numpy()

            # MovieLens ratings are between 0.5 and 5.0
            predictions = np.clip(predictions, 0.5, 5.0)

            all_predictions.extend(predictions)
            all_targets.extend(ratings)

    rmse = calculate_rmse(all_targets, all_predictions)
    mae = calculate_mae(all_targets, all_predictions)

    return rmse, mae


def train_model(model, train_loader, valid_loader, device):
    """
    Train Neural Collaborative Filtering model.
    """

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    train_losses = []
    valid_rmse_values = []
    valid_mae_values = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for users, movies, ratings in train_loader:
            users = users.to(device)
            movies = movies.to(device)
            ratings = ratings.to(device)

            optimizer.zero_grad()

            predictions = model(users, movies)
            loss = criterion(predictions, ratings)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_train_loss = total_loss / len(train_loader)

        valid_rmse, valid_mae = evaluate_model(
            model,
            valid_loader,
            device
        )

        train_losses.append(average_train_loss)
        valid_rmse_values.append(valid_rmse)
        valid_mae_values.append(valid_mae)

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {average_train_loss:.4f}")
        print(f"Validation RMSE: {valid_rmse:.4f}")
        print(f"Validation MAE: {valid_mae:.4f}")
        print("-" * 40)

    history_df = pd.DataFrame({
        "epoch": list(range(1, EPOCHS + 1)),
        "train_loss": train_losses,
        "valid_rmse": valid_rmse_values,
        "valid_mae": valid_mae_values
    })

    return history_df


def save_results(test_rmse, test_mae, history_df):
    """
    Save training history and final test results.
    """

    os.makedirs(RESULTS_DIR, exist_ok=True)

    deep_results = pd.DataFrame([
        {
            "model": "Neural Collaborative Filtering",
            "RMSE": test_rmse,
            "MAE": test_mae
        }
    ])

    deep_results_csv = os.path.join(
        RESULTS_DIR,
        "week3_deep_learning_results.csv"
    )

    deep_results_txt = os.path.join(
        RESULTS_DIR,
        "week3_deep_learning_results.txt"
    )

    history_csv = os.path.join(
        RESULTS_DIR,
        "week3_training_history.csv"
    )

    deep_results.to_csv(deep_results_csv, index=False)
    history_df.to_csv(history_csv, index=False)

    with open(deep_results_txt, "w", encoding="utf-8") as file:
        file.write("Week 3 Deep Learning Model Results\n")
        file.write("=" * 40 + "\n\n")
        file.write(deep_results.to_string(index=False))

    print("Results saved:")
    print(deep_results_csv)
    print(deep_results_txt)
    print(history_csv)


def compare_with_baseline(deep_results_path):
    """
    Compare Week 3 deep learning results with Week 2 baseline results.
    """

    baseline_path = os.path.join(
        RESULTS_DIR,
        "week2_baseline_results.csv"
    )

    if not os.path.exists(baseline_path):
        print("Week 2 baseline results not found.")
        return

    baseline_results = pd.read_csv(baseline_path)
    deep_results = pd.read_csv(deep_results_path)

    comparison_df = pd.concat(
        [baseline_results, deep_results],
        ignore_index=True
    )

    comparison_path = os.path.join(
        RESULTS_DIR,
        "week3_model_comparison.csv"
    )

    comparison_df.to_csv(comparison_path, index=False)

    print("Model comparison saved:")
    print(comparison_path)


def save_model(model, num_users, num_movies):
    """
    Save trained PyTorch model.
    """

    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(
        MODELS_DIR,
        "week3_neural_collaborative_filtering.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_users": num_users,
            "num_movies": num_movies,
            "embedding_dim": EMBEDDING_DIM
        },
        model_path
    )

    print("Model saved:")
    print(model_path)


def main():
    print("Week 3: Neural Collaborative Filtering")
    print("=" * 45)

    device = get_device()
    print("Device:", device)

    print("\nLoading processed data...")
    train_df, valid_df, test_df = load_processed_data()

    print("Train shape:", train_df.shape)
    print("Validation shape:", valid_df.shape)
    print("Test shape:", test_df.shape)

    num_users, num_movies = get_number_of_users_and_movies(
        train_df,
        valid_df,
        test_df
    )

    print("\nNumber of users:", num_users)
    print("Number of movies:", num_movies)

    print("\nCreating DataLoaders...")
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_df,
        valid_df,
        test_df
    )

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(valid_loader))
    print("Test batches:", len(test_loader))

    print("\nBuilding model...")
    model = NeuralCollaborativeFiltering(
        num_users=num_users,
        num_movies=num_movies,
        embedding_dim=EMBEDDING_DIM
    ).to(device)

    print(model)

    print("\nTraining model...")
    history_df = train_model(
        model,
        train_loader,
        valid_loader,
        device
    )

    print("\nEvaluating model on test set...")
    test_rmse, test_mae = evaluate_model(
        model,
        test_loader,
        device
    )

    print("\nFinal Test Results")
    print("==================")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE:  {test_mae:.4f}")

    print("\nSaving results...")
    save_results(
        test_rmse,
        test_mae,
        history_df
    )

    deep_results_path = os.path.join(
        RESULTS_DIR,
        "week3_deep_learning_results.csv"
    )

    print("\nComparing with baseline...")
    compare_with_baseline(deep_results_path)

    print("\nSaving model...")
    save_model(
        model,
        num_users,
        num_movies
    )

    print("\nWeek 3 deep learning model completed successfully.")


if __name__ == "__main__":
    main()
