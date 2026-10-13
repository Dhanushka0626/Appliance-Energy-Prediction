"""Data preprocessing for the Appliance Energy Prediction project."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "Appliances"
EXPECTED_INTERVAL = pd.Timedelta(minutes=10)
TRAIN_FRACTION = 0.80


def load_data(csv_path):
    """Load the raw dataset from a CSV file."""
    return pd.read_csv(csv_path)


def clean_data(df):
    """Parse timestamps, drop invalid rows and sort chronologically."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def check_data_quality(df):
    """Return a summary of missing values, duplicates and timestamp gaps."""
    intervals = df["date"].diff().dropna()
    return {
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "repeated_timestamps": int(df["date"].duplicated().sum()),
        "negative_target_values": int((df[TARGET_COLUMN] < 0).sum()),
        "unexpected_intervals": int((intervals != EXPECTED_INTERVAL).sum()),
        "chronologically_sorted": bool(df["date"].is_monotonic_increasing),
    }


def chronological_split(df, train_fraction=TRAIN_FRACTION):
    """Split chronologically (no shuffling) into train and test sets."""
    split_index = int(len(df) * train_fraction)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def iqr_outlier_bounds(series):
    """Return (lower, upper) IQR outlier bounds for a series."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def get_feature_columns(df, excluded=("date", TARGET_COLUMN, "hour", "day_of_week")):
    """Select numeric feature columns, excluding target, timestamp and EDA-only columns."""
    return [
        col for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


def scale_features(train_df, test_df, feature_columns):
    """Fit StandardScaler on training features only and apply it to both sets."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[feature_columns])
    X_test_scaled = scaler.transform(test_df[feature_columns])
    return X_train_scaled, X_test_scaled, scaler


def persistence_forecast(train_df, test_df):
    """Predict each test value with the previous observed target value."""
    previous = pd.concat(
        [train_df[TARGET_COLUMN].tail(1), test_df[TARGET_COLUMN].iloc[:-1]],
        ignore_index=True,
    )
    return previous.to_numpy()
