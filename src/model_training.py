"""
Model Training Module
======================
Handles train-test splitting, feature scaling, and training of
Random Forest Regressor and SVR models for the USD/LKR pipeline.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


# ---------------------------------------------------------------------------
# 1. Time-based train-test split (NO random shuffling)
# ---------------------------------------------------------------------------
def time_based_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "Target",
    train_ratio: float = 0.80,
) -> tuple:
    """
    Split the data chronologically — first 80 % for training, last 20 % for testing.

    Args:
        df: Feature-engineered DataFrame (sorted by date).
        feature_cols: List of feature column names.
        target_col: Name of the target column.
        train_ratio: Proportion of data used for training.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, train_index, test_index)
    """
    split_idx = int(len(df) * train_ratio)

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    train_index = df.index[:split_idx]
    test_index = df.index[split_idx:]

    print(f"[INFO] Time-based split — Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, train_index, test_index


# ---------------------------------------------------------------------------
# 2. Feature scaling (StandardScaler for SVR)
# ---------------------------------------------------------------------------
def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple:
    """
    Fit a StandardScaler on training data and transform both splits.

    Args:
        X_train: Training feature matrix.
        X_test: Testing feature matrix.

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Persist scaler for future inference
    os.makedirs(MODELS_DIR, exist_ok=True)
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] StandardScaler saved to {scaler_path}")

    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# 3. Train Random Forest Regressor
# ---------------------------------------------------------------------------
def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    random_state: int = 42,
) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor and save it to disk.

    Args:
        X_train: Training feature matrix (unscaled is fine for tree models).
        y_train: Training target array.
        n_estimators: Number of trees.
        random_state: Seed for reproducibility.

    Returns:
        Trained RandomForestRegressor instance.
    """
    print("[INFO] Training Random Forest Regressor …")
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    print("[INFO] Random Forest training complete.")

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "random_forest_model.pkl")
    joblib.dump(rf_model, model_path)
    print(f"[INFO] Model saved to {model_path}")

    return rf_model


# ---------------------------------------------------------------------------
# 4. Train Support Vector Regressor (RBF kernel)
# ---------------------------------------------------------------------------
def train_svr(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    kernel: str = "rbf",
    C: float = 100.0,
    epsilon: float = 0.1,
) -> SVR:
    """
    Train a Support Vector Regressor with an RBF kernel and save it.

    Args:
        X_train_scaled: Scaled training feature matrix.
        y_train: Training target array.
        kernel: SVM kernel type.
        C: Regularisation parameter.
        epsilon: Epsilon-tube within which no penalty is applied.

    Returns:
        Trained SVR instance.
    """
    print("[INFO] Training SVR (RBF kernel) …")
    svr_model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    svr_model.fit(X_train_scaled, y_train)
    print("[INFO] SVR training complete.")

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "svr_model.pkl")
    joblib.dump(svr_model, model_path)
    print(f"[INFO] Model saved to {model_path}")

    return svr_model
