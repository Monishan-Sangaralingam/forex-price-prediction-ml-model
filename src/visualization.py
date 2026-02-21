"""
Visualization Module
=====================
Generates plots for model predictions and feature importance,
saving them to the results/ directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


# ---------------------------------------------------------------------------
# 1. Actual vs Predicted — line chart for both models
# ---------------------------------------------------------------------------
def plot_actual_vs_predicted(
    test_index: pd.DatetimeIndex,
    y_true: np.ndarray,
    rf_pred: np.ndarray,
    svr_pred: np.ndarray,
    save_path: str | None = None,
) -> str:
    """
    Plot Actual vs Predicted values for Random Forest and SVR on the test set.

    Args:
        test_index: DatetimeIndex of the test period.
        y_true: Actual target values.
        rf_pred: Random Forest predictions.
        svr_pred: SVR predictions.
        save_path: File path to save the figure. Defaults to results/actual_vs_predicted.png.

    Returns:
        Path to the saved figure.
    """
    if save_path is None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_path = os.path.join(RESULTS_DIR, "actual_vs_predicted.png")

    plt.figure(figsize=(14, 6))
    plt.plot(test_index, y_true, label="Actual", color="black", linewidth=2)
    plt.plot(test_index, rf_pred, label="Random Forest", color="steelblue", linewidth=1.5)
    plt.plot(test_index, svr_pred, label="SVR", color="coral", linewidth=1.5)

    plt.title("USD/LKR Next-Day Close — Actual vs Predicted", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Exchange Rate (LKR per USD)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"[INFO] Actual vs Predicted plot saved to {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# 2. Feature Importance — bar chart from Random Forest
# ---------------------------------------------------------------------------
def plot_feature_importance(
    rf_model: RandomForestRegressor,
    feature_names: list[str],
    save_path: str | None = None,
) -> str:
    """
    Display and save a horizontal bar chart of feature importances from the
    Random Forest model.

    Args:
        rf_model: Trained RandomForestRegressor.
        feature_names: List of feature column names (in the same order used for training).
        save_path: File path to save the figure. Defaults to results/feature_importance.png.

    Returns:
        Path to the saved figure.
    """
    if save_path is None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_path = os.path.join(RESULTS_DIR, "feature_importance.png")

    importances = rf_model.feature_importances_
    indices = np.argsort(importances)

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    plt.barh(
        range(len(indices)),
        importances[indices],
        color="steelblue",
        edgecolor="black",
    )
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=11)
    plt.xlabel("Importance", fontsize=12)
    plt.title("Random Forest — Feature Importance", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    # Print to console as well
    print("\n" + "=" * 45)
    print("  Random Forest — Feature Importance")
    print("=" * 45)
    for i in reversed(indices):
        print(f"  {feature_names[i]:<20} {importances[i]:.4f}")
    print("=" * 45 + "\n")

    print(f"[INFO] Feature importance plot saved to {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# 3. Model comparison bar chart
# ---------------------------------------------------------------------------
def plot_metrics_comparison(
    rf_metrics: dict,
    svr_metrics: dict,
    save_path: str | None = None,
) -> str:
    """
    Side-by-side bar chart comparing MAE, RMSE, and R² for both models.

    Args:
        rf_metrics: Dict with keys MAE, RMSE, R2 for Random Forest.
        svr_metrics: Dict with keys MAE, RMSE, R2 for SVR.
        save_path: Optional file path. Defaults to results/metrics_comparison.png.

    Returns:
        Path to the saved figure.
    """
    if save_path is None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_path = os.path.join(RESULTS_DIR, "metrics_comparison.png")

    metrics = ["MAE", "RMSE", "R2"]
    rf_vals = [rf_metrics[m] for m in metrics]
    svr_vals = [svr_metrics[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, rf_vals, width, label="Random Forest", color="steelblue")
    bars2 = ax.bar(x + width / 2, svr_vals, width, label="SVR", color="coral")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Evaluation Metrics", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(["MAE", "RMSE", "R²"], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"[INFO] Metrics comparison plot saved to {save_path}")
    return save_path
