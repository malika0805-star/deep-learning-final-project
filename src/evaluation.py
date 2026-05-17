import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error.

    RMSE gives stronger penalty to large prediction errors.
    Lower RMSE means better model performance.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error.

    MAE shows the average absolute difference
    between real and predicted ratings.
    Lower MAE means better model performance.
    """
    return mean_absolute_error(y_true, y_pred)


def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluate rating prediction model using RMSE and MAE.
    """
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)

    print(model_name)
    print("=" * len(model_name))
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")

    return {
        "model": model_name,
        "RMSE": rmse,
        "MAE": mae
    }


def save_results(results_df, output_txt_path):
    """
    Save model results into a text file.
    """
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

    with open(output_txt_path, "w", encoding="utf-8") as file:
        file.write("Model Evaluation Results\n")
        file.write("=" * 30 + "\n\n")
        file.write(results_df.to_string(index=False))

    print("Results saved to:", output_txt_path)
