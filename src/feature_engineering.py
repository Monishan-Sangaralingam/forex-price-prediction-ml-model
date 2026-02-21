"""
Feature Engineering Module
===========================
Creates technical indicator features and the target variable
for the USD/LKR Forex prediction pipeline.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# 1. Create technical-indicator features
# ---------------------------------------------------------------------------
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate new features from the Close price series.

    Features created:
        - MA_5   : 5-day simple moving average
        - MA_10  : 10-day simple moving average
        - ROC    : Rate of Change (daily % change)
        - High_Low_Diff  : (not available directly; approximated as
                            daily absolute change for this single-column dataset)
        - Open_Close_Diff: (approximated as Close − previous Close)

    Note:
        The raw Kaggle dataset provides only one price column per currency.
        High, Low, and Open are not available, so we approximate those
        features using Close-only calculations that still capture volatility
        and momentum signals.

    Args:
        df: DataFrame with DatetimeIndex and a 'Close' column.

    Returns:
        DataFrame with new feature columns appended.
    """
    df = df.copy()

    # Moving averages
    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["MA_10"] = df["Close"].rolling(window=10).mean()

    # Rate of Change (percentage change from previous day)
    df["ROC"] = df["Close"].pct_change() * 100

    # High-Low difference approximation (absolute daily change)
    df["High_Low_Diff"] = df["Close"].diff().abs()

    # Open-Close difference approximation (Close − previous Close)
    df["Open_Close_Diff"] = df["Close"].diff()

    print(f"[INFO] Features created: MA_5, MA_10, ROC, High_Low_Diff, Open_Close_Diff")
    return df


# ---------------------------------------------------------------------------
# 2. Create target variable
# ---------------------------------------------------------------------------
def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the target variable: next-day Close price.

    Target = Close.shift(-1)

    Args:
        df: DataFrame that must contain a 'Close' column.

    Returns:
        DataFrame with an additional 'Target' column.
    """
    df = df.copy()
    df["Target"] = df["Close"].shift(-1)
    print("[INFO] Target variable created: Close.shift(-1)")
    return df


# ---------------------------------------------------------------------------
# 3. Remove NaN rows introduced by rolling / shift operations
# ---------------------------------------------------------------------------
def drop_na_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop any rows that contain NaN values (from rolling windows and target shift).

    Args:
        df: DataFrame potentially containing NaN values.

    Returns:
        Cleaned DataFrame with no NaN values.
    """
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"[INFO] Dropped {before - after} NaN rows → {after} rows remaining")
    return df


# ---------------------------------------------------------------------------
# Pipeline convenience function
# ---------------------------------------------------------------------------
def run_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the full feature engineering pipeline:
        create features → create target → drop NaN rows

    Args:
        df: Cleaned DataFrame with DatetimeIndex and 'Close' column.

    Returns:
        Feature-rich DataFrame ready for modelling.
    """
    df = create_features(df)
    df = create_target(df)
    df = drop_na_rows(df)
    print(f"[INFO] Final feature set columns: {list(df.columns)}")
    return df
