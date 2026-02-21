"""
Data Preprocessing Module
=========================
Handles dataset downloading, loading, cleaning, and filtering
for the USD/LKR Forex prediction pipeline.
"""

import os
import shutil
import pandas as pd
import kagglehub


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_FILENAME = "Foreign_Exchange_Rates.csv"
DATASET_SLUG = "thebasss/currency-exchange-rates"
TARGET_CURRENCY = "SRI LANKA - SRI LANKAN RUPEE/US$"


# ---------------------------------------------------------------------------
# 1. Download dataset via kagglehub
# ---------------------------------------------------------------------------
def download_dataset() -> str:
    """
    Download the Currency Exchange Rates dataset from Kaggle using kagglehub.

    Returns:
        str: Path to the downloaded CSV file inside the project's data/ folder.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    local_path = os.path.join(DATA_DIR, RAW_FILENAME)

    if os.path.exists(local_path):
        print(f"[INFO] Dataset already exists at {local_path}")
        return local_path

    print("[INFO] Downloading dataset from Kaggle …")
    downloaded_dir = kagglehub.dataset_download(DATASET_SLUG)
    print(f"[INFO] Downloaded to: {downloaded_dir}")

    # kagglehub returns a directory; locate the CSV inside it
    src_file = None
    for root, _, files in os.walk(downloaded_dir):
        for f in files:
            if f.endswith(".csv"):
                src_file = os.path.join(root, f)
                break
        if src_file:
            break

    if src_file is None:
        raise FileNotFoundError("CSV file not found inside the downloaded dataset directory.")

    shutil.copy2(src_file, local_path)
    print(f"[INFO] Copied dataset to {local_path}")
    return local_path


# ---------------------------------------------------------------------------
# 2. Load raw dataset
# ---------------------------------------------------------------------------
def load_raw_data(filepath: str | None = None) -> pd.DataFrame:
    """
    Load the raw CSV file into a pandas DataFrame.

    Args:
        filepath: Optional explicit path. If None, uses DATA_DIR/RAW_FILENAME.

    Returns:
        pd.DataFrame with raw exchange-rate data.
    """
    if filepath is None:
        filepath = os.path.join(DATA_DIR, RAW_FILENAME)

    print(f"[INFO] Loading raw data from {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Raw data shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 3. Display dataset summary
# ---------------------------------------------------------------------------
def show_data_summary(df: pd.DataFrame) -> None:
    """Print useful summary statistics about the DataFrame."""
    print("\n" + "=" * 50)
    print("  Dataset Summary")
    print("=" * 50)
    print(f"\nShape : {df.shape}")
    print(f"Columns ({len(df.columns)}):\n  {list(df.columns)}")
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())
    print("\nBasic statistics:")
    print(df.describe().to_string())
    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# 4. Filter for USD/LKR currency pair
# ---------------------------------------------------------------------------
def filter_usd_lkr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the USD/LKR exchange rate data from the wide-format dataset.

    The Kaggle dataset stores currencies as separate columns.  We select only
    the Date and the Sri Lankan Rupee column, then rename for clarity.

    Args:
        df: Raw wide-format DataFrame.

    Returns:
        pd.DataFrame with columns: Date, Close
    """
    date_col = df.columns[0]  # first column is the date (usually unnamed or "Time Serie")

    if TARGET_CURRENCY not in df.columns:
        # Some dataset versions may have slightly different names
        matches = [c for c in df.columns if "SRI LANKA" in c.upper() or "LKR" in c.upper()]
        if matches:
            currency_col = matches[0]
        else:
            raise KeyError(
                f"Column '{TARGET_CURRENCY}' not found. Available: {list(df.columns)}"
            )
    else:
        currency_col = TARGET_CURRENCY

    filtered = df[[date_col, currency_col]].copy()
    filtered.columns = ["Date", "Close"]
    print(f"[INFO] Filtered USD/LKR data — shape: {filtered.shape}")
    return filtered


# ---------------------------------------------------------------------------
# 5. Clean and prepare the filtered data
# ---------------------------------------------------------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the filtered USD/LKR DataFrame:
        - Convert Date to datetime
        - Convert Close to numeric (coerce errors → NaN)
        - Drop rows with missing values
        - Sort chronologically
        - Set Date as the index

    Args:
        df: DataFrame with columns Date, Close.

    Returns:
        Cleaned and indexed pd.DataFrame.
    """
    df = df.copy()

    # Convert types
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # Drop any rows that couldn't be parsed
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    if before != after:
        print(f"[INFO] Dropped {before - after} rows with missing/invalid values")

    # Sort by date and set as index
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)

    print(f"[INFO] Preprocessed data shape: {df.shape}")
    print(f"[INFO] Date range: {df.index.min()} → {df.index.max()}")
    return df


# ---------------------------------------------------------------------------
# Pipeline convenience function
# ---------------------------------------------------------------------------
def run_data_pipeline() -> pd.DataFrame:
    """
    Execute the full data pipeline:
        download → load → summarise → filter → preprocess

    Returns:
        Cleaned pd.DataFrame ready for feature engineering.
    """
    path = download_dataset()
    raw = load_raw_data(path)
    show_data_summary(raw)
    filtered = filter_usd_lkr(raw)
    cleaned = preprocess_data(filtered)
    return cleaned
