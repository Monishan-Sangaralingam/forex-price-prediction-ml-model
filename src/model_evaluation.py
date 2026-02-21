"""
Model Evaluation Module
========================
Calculates regression metrics (MAE, RMSE, R²) for each model
and persists the results to results/metrics.txt.
"""

import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


# ---------------------------------------------------------------------------
# 1. Calculate evaluation metrics
# ---------------------------------------------------------------------------
def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> dict:
    """
    Compute MAE, RMSE, and R² score for a given set of predictions.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted values.
        model_name: Label used for display.

    Returns:
        Dictionary with keys 'MAE', 'RMSE', 'R2'.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{'=' * 45}")
    print(f"  {model_name} — Evaluation Metrics")
    print(f"{'=' * 45}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    print(f"{'=' * 45}\n")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ---------------------------------------------------------------------------
# 2. Save metrics to a text file
# ---------------------------------------------------------------------------
def save_metrics(
    rf_metrics: dict,
    svr_metrics: dict,
    filepath: str | None = None,
) -> str:
    """
    Write the evaluation results of both models to a plain-text file.

    Args:
        rf_metrics: Metrics dict for Random Forest.
        svr_metrics: Metrics dict for SVR.
        filepath: Destination file. Defaults to results/metrics.txt.

    Returns:
        Path to the saved metrics file.
    """
    if filepath is None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        filepath = os.path.join(RESULTS_DIR, "metrics.txt")

    lines = [
        "=" * 55,
        "  Forex USD/LKR Prediction — Model Evaluation Results",
        "=" * 55,
        "",
        "-" * 55,
        f"  {'Metric':<10} {'Random Forest':>15} {'SVR':>15}",
        "-" * 55,
        f"  {'MAE':<10} {rf_metrics['MAE']:>15.4f} {svr_metrics['MAE']:>15.4f}",
        f"  {'RMSE':<10} {rf_metrics['RMSE']:>15.4f} {svr_metrics['RMSE']:>15.4f}",
        f"  {'R²':<10} {rf_metrics['R2']:>15.4f} {svr_metrics['R2']:>15.4f}",
        "-" * 55,
        "",
        "Best model by R²: "
        + ("Random Forest" if rf_metrics["R2"] >= svr_metrics["R2"] else "SVR"),
        "",
    ]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[INFO] Metrics saved to {filepath}")
    return filepath
