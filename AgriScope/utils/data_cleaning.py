"""
AgriScope - Data Cleaning Module
Cleans and preprocesses final_data.csv for ML training.
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded data with shape: {df.shape}")
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase with underscores."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    print(f"[INFO] Standardized columns: {list(df.columns)}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"[INFO] Removed {before - after} duplicate rows.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values: median for numerical, mode for categorical."""
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64, float, int]:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
    print("[INFO] Missing values handled.")
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert datatypes: district → string, season → category."""
    if "district" in df.columns:
        df["district"] = df["district"].astype(str)
    if "season" in df.columns:
        df["season"] = df["season"].astype("category")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create yield_calculated = production / area."""
    if "production" in df.columns and "area" in df.columns:
        df["yield_calculated"] = df.apply(
            lambda row: row["production"] / row["area"] if row["area"] > 0 else 0,
            axis=1
        )
        print("[INFO] Feature 'yield_calculated' created.")
    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Remove outliers using IQR method for specified columns."""
    initial_len = len(df)
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    removed = initial_len - len(df)
    print(f"[INFO] Removed {removed} outlier rows.")
    return df


def encode_categorical(df: pd.DataFrame) -> tuple:
    """
    Label-encode district, season, and crop_type.
    Returns (df, encoders_dict).
    """
    encoders = {}
    for col in ["district", "season", "crop_type"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"[INFO] Encoded '{col}' → '{col}_encoded'")
    return df, encoders


def clean_data(raw_path: str, output_path: str) -> pd.DataFrame:
    """
    Full pipeline: load → standardize → dedup → fill missing →
    fix dtypes → feature engineering → outlier removal → encode → save.
    """
    df = load_data(raw_path)
    df = standardize_columns(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = fix_dtypes(df)
    df = feature_engineering(df)

    # Drop rows where yield is 0 (no production)
    if "yield" in df.columns:
        df = df[df["yield"] > 0].copy()

    # Outlier removal on key numeric columns
    outlier_cols = ["total_rainfall", "yield", "production", "area"]
    existing_outlier_cols = [c for c in outlier_cols if c in df.columns]
    df = remove_outliers_iqr(df, existing_outlier_cols)

    df, encoders = encode_categorical(df)

    # Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Cleaned data saved to: {output_path}")
    print(f"[INFO] Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, "data", "final_data.csv")
    output_path = os.path.join(base_dir, "cleaned_data", "cleaned_data.csv")
    clean_data(raw_path, output_path)
