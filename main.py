"""
Forex Price Prediction ML Model
================================
Main entry point for the USD/LKR exchange rate prediction pipeline.

This script orchestrates the complete ML pipeline:
1. Data loading and preprocessing
2. Feature engineering
3. Train-test splitting
4. Feature scaling
5. Model training (Random Forest & SVR)
6. Model evaluation
7. Visualization and feature importance analysis

Usage:
    python main.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Ensure the project root is on the path so src imports work
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import run_data_pipeline
from src.feature_engineering import run_feature_pipeline
from src.model_training import (
    time_based_split,
    scale_features,
    train_random_forest,
    train_svr,
)
from src.model_evaluation import evaluate_model, save_metrics
from src.visualization import (
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_metrics_comparison,
)


# Feature columns used for modelling
FEATURE_COLS = ["Close", "MA_5", "MA_10", "ROC", "High_Low_Diff", "Open_Close_Diff"]


def main():
    """Run the complete ML pipeline."""
    print("=" * 60)
    print("  Forex Price Prediction — USD/LKR")
    print("  Random Forest Regressor vs Support Vector Regressor")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1-3: Data loading & preprocessing
    # ------------------------------------------------------------------
    print("\n>>> STAGE 1: Data Loading & Preprocessing")
    df = run_data_pipeline()

    # ------------------------------------------------------------------
    # Step 4: Feature engineering
    # ------------------------------------------------------------------
    print("\n>>> STAGE 2: Feature Engineering")
    df = run_feature_pipeline(df)

    # ------------------------------------------------------------------
    # Step 5: Time-based train-test split
    # ------------------------------------------------------------------
    print("\n>>> STAGE 3: Train-Test Split (80/20 chronological)")
    X_train, X_test, y_train, y_test, train_idx, test_idx = time_based_split(
        df, FEATURE_COLS
    )

    # ------------------------------------------------------------------
    # Step 6: Feature scaling (for SVR)
    # ------------------------------------------------------------------
    print("\n>>> STAGE 4: Feature Scaling")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # ------------------------------------------------------------------
    # Step 7: Train Random Forest Regressor
    # ------------------------------------------------------------------
    print("\n>>> STAGE 5: Training Random Forest Regressor")
    rf_model = train_random_forest(X_train, y_train)

    # ------------------------------------------------------------------
    # Step 8: Train SVR (RBF kernel)
    # ------------------------------------------------------------------
    print("\n>>> STAGE 6: Training Support Vector Regressor")
    svr_model = train_svr(X_train_scaled, y_train)

    # ------------------------------------------------------------------
    # Step 9: Model evaluation
    # ------------------------------------------------------------------
    print("\n>>> STAGE 7: Model Evaluation")
    rf_pred = rf_model.predict(X_test)
    svr_pred = svr_model.predict(X_test_scaled)

    rf_metrics = evaluate_model(y_test, rf_pred, "Random Forest Regressor")
    svr_metrics = evaluate_model(y_test, svr_pred, "Support Vector Regressor")

    save_metrics(rf_metrics, svr_metrics)

    # ------------------------------------------------------------------
    # Step 10: Visualization — Actual vs Predicted
    # ------------------------------------------------------------------
    print("\n>>> STAGE 8: Generating Plots")
    plot_actual_vs_predicted(test_idx, y_test, rf_pred, svr_pred)
    plot_metrics_comparison(rf_metrics, svr_metrics)

    # ------------------------------------------------------------------
    # Step 11: Feature importance (Random Forest)
    # ------------------------------------------------------------------
    print("\n>>> STAGE 9: Feature Importance Analysis")
    plot_feature_importance(rf_model, FEATURE_COLS)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("  • Models saved in   : models/")
    print("  • Metrics saved in  : results/metrics.txt")
    print("  • Plots saved in    : results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
