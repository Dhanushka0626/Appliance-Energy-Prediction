"""Data preprocessing for the Appliance Energy Prediction project.

Mirrors section 1 of the notebook and the data preparation of section 3.1:
loading, cleaning, quality checks, outlier analysis, the chronological
80 % / 20 % train/test split, a validation split taken from the training part
only, leakage-safe scaling and sequence windows, and evaluation helpers.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "Appliances"
EXPECTED_INTERVAL = pd.Timedelta(minutes=10)
TRAIN_FRACTION = 0.80   # chronological train / test split
VAL_FRACTION = 0.15     # share of the TRAINING part kept for validation


# ---------------------------------------------------------------------------
# Loading, cleaning, quality checks
# ---------------------------------------------------------------------------
def load_data(csv_path):
    """Load the raw dataset from a CSV file."""
    return pd.read_csv(csv_path)


def clean_data(df):
    """Parse timestamps, sort chronologically and handle missing values.

    Rows with an invalid timestamp or a missing target are dropped. Missing
    sensor readings (short gaps only) are filled by time-based interpolation;
    the target is never imputed.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if df.isna().sum().sum() > 0:
        sensor_cols = [c for c in df.columns if c not in ("date", TARGET_COLUMN)]
        df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
        df[sensor_cols] = (
            df.set_index("date")[sensor_cols].interpolate(method="time", limit=6).values
        )
    return df


def check_data_quality(df):
    """Return a summary of missing values, duplicates and timestamp problems."""
    intervals = df["date"].diff().dropna()
    return {
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "repeated_timestamps": int(df["date"].duplicated().sum()),
        "negative_target_values": int((df[TARGET_COLUMN] < 0).sum()),
        "unexpected_intervals": int((intervals != EXPECTED_INTERVAL).sum()),
        "chronologically_sorted": bool(df["date"].is_monotonic_increasing),
    }


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------
def iqr_outlier_bounds(series):
    """Return (lower, upper) IQR outlier bounds for a series."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def zscore_outlier_mask(series, threshold=3.0):
    """Boolean mask of values whose |z-score| exceeds the threshold."""
    return ((series - series.mean()) / series.std()).abs() > threshold


# ---------------------------------------------------------------------------
# Chronological splitting (never shuffled)
# ---------------------------------------------------------------------------
def chronological_split(df, train_fraction=TRAIN_FRACTION):
    """Split into the first 80 % (training) and last 20 % (testing)."""
    split_index = int(len(df) * train_fraction)
    return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()


def build_timeline(model_df, feature_pool, split_index, val_fraction=VAL_FRACTION):
    """One chronological timeline: training sub-set | validation | test.

    The 80 % / 20 % boundary (`split_index`) is applied first. Rows lacking
    lag/rolling history are dropped inside each part. The validation set is
    the last `val_fraction` of the TRAINING part only; the test part is never
    touched. Returns (full, idx_sub, idx_val, idx_test).
    """
    required = list(feature_pool) + [TARGET_COLUMN]
    train_ready = model_df.iloc[:split_index].dropna(subset=required)
    test_ready = model_df.iloc[split_index:].dropna(subset=required)

    full = pd.concat([train_ready, test_ready], ignore_index=True)
    n_train, n_test = len(train_ready), len(test_ready)
    n_val = int(n_train * val_fraction)
    n_sub = n_train - n_val

    idx_sub = np.arange(0, n_sub)
    idx_val = np.arange(n_sub, n_train)
    idx_test = np.arange(n_train, n_train + n_test)

    assert (full["date"].diff().dropna() == EXPECTED_INTERVAL).all(), "timeline must be contiguous"
    return full, idx_sub, idx_val, idx_test


# ---------------------------------------------------------------------------
# Scaling and sequence windows
# ---------------------------------------------------------------------------
def fit_scalers(full, features, idx_sub):
    """Fit StandardScalers on the training sub-set only; return scaled arrays too."""
    y = full[TARGET_COLUMN].to_numpy(dtype=np.float32)
    x_scaler = StandardScaler().fit(full.loc[idx_sub, features])
    y_scaler = StandardScaler().fit(y[idx_sub].reshape(-1, 1))
    X_scaled = x_scaler.transform(full[features]).astype(np.float32)
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).astype(np.float32).ravel()
    return x_scaler, y_scaler, X_scaled, y_scaled


def make_windows(X_scaled, y_scaled, idx, lookback):
    """Windows for target rows `idx`: rows i-lookback+1 ... i (the target row included).

    Row i holds only information available before interval i (lags, rolling
    statistics, previous-interval sensors), so it must be part of the window:
    it carries the most recent consumption value.
    """
    idx = np.asarray(idx)
    idx = idx[idx >= lookback - 1]
    X = np.stack([X_scaled[i - lookback + 1: i + 1] for i in idx]).astype(np.float32)
    return X, y_scaled[idx], idx


# ---------------------------------------------------------------------------
# Evaluation helpers (always in Wh)
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred):
    """MAE, RMSE, MAPE and R2 in the original Wh scale."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    err = y_true - y_pred
    nonzero = y_true != 0
    return {
        "MAE (Wh)": mean_absolute_error(y_true, y_pred),
        "RMSE (Wh)": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE (%)": float(np.mean(np.abs(err[nonzero] / y_true[nonzero])) * 100),
        "R²": r2_score(y_true, y_pred),
    }


def persistence_forecast(full, idx):
    """Predict each interval with the previous interval's consumption (appliances_lag_1)."""
    return full.loc[idx, "appliances_lag_1"].to_numpy()
