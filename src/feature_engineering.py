"""Feature engineering for the Appliance Energy Prediction project."""

import numpy as np
import pandas as pd

TARGET_COLUMN = "Appliances"
EXCLUDED_COLUMNS = {"date", TARGET_COLUMN, "hour", "day_of_week", "month"}


def prepare_base_frame(df):
    """Parse dates, sort chronologically and drop duplicate timestamps."""
    model_df = df.copy()
    model_df["date"] = pd.to_datetime(model_df["date"], errors="coerce")
    return (
        model_df.dropna(subset=["date"])
                .sort_values("date")
                .drop_duplicates(subset="date")
                .reset_index(drop=True)
    )


def add_calendar_features(model_df):
    """Add calendar features and their cyclical (sin/cos) encodings."""
    model_df = model_df.copy()
    model_df["hour"] = model_df["date"].dt.hour
    model_df["day_of_week"] = model_df["date"].dt.dayofweek
    model_df["month"] = model_df["date"].dt.month
    model_df["is_weekend"] = (model_df["day_of_week"] >= 5).astype(int)

    model_df["hour_sin"] = np.sin(2 * np.pi * model_df["hour"] / 24)
    model_df["hour_cos"] = np.cos(2 * np.pi * model_df["hour"] / 24)
    model_df["day_sin"] = np.sin(2 * np.pi * model_df["day_of_week"] / 7)
    model_df["day_cos"] = np.cos(2 * np.pi * model_df["day_of_week"] / 7)
    model_df["month_sin"] = np.sin(2 * np.pi * (model_df["month"] - 1) / 12)
    model_df["month_cos"] = np.cos(2 * np.pi * (model_df["month"] - 1) / 12)
    return model_df


def add_rolling_features(model_df):
    """Add rolling statistics computed from past target values only."""
    model_df = model_df.copy()
    previous = model_df[TARGET_COLUMN].shift(1)
    model_df["appliances_rolling_mean_1h"] = previous.rolling(6, min_periods=6).mean()
    model_df["appliances_rolling_mean_3h"] = previous.rolling(18, min_periods=18).mean()
    model_df["appliances_rolling_std_1h"] = previous.rolling(6, min_periods=6).std()
    return model_df


def add_lag_features(model_df, lags=(1, 3, 6, 18)):
    """Add lagged target features (10-minute steps)."""
    model_df = model_df.copy()
    for lag in lags:
        model_df[f"appliances_lag_{lag}"] = model_df[TARGET_COLUMN].shift(lag)
    return model_df


def add_interaction_features(model_df):
    """Add temperature x humidity interactions when source columns exist."""
    model_df = model_df.copy()
    if {"T2", "RH_2"}.issubset(model_df.columns):
        model_df["T2_RH2_interaction"] = model_df["T2"] * model_df["RH_2"]
    if {"T_out", "RH_out"}.issubset(model_df.columns):
        model_df["Tout_RHout_interaction"] = model_df["T_out"] * model_df["RH_out"]
    return model_df


def build_features(df):
    """Run the full feature engineering pipeline."""
    model_df = prepare_base_frame(df)
    model_df = add_calendar_features(model_df)
    model_df = add_rolling_features(model_df)
    model_df = add_lag_features(model_df)
    model_df = add_interaction_features(model_df)
    return model_df


def get_candidate_features(model_df):
    """Return numeric candidate feature columns."""
    return [
        col for col in model_df.columns
        if col not in EXCLUDED_COLUMNS and pd.api.types.is_numeric_dtype(model_df[col])
    ]


def split_and_clean(model_df, train_fraction=0.80):
    """Chronological split, then drop rows lacking lag/rolling history in each period."""
    model_df = model_df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(model_df) * train_fraction)
    features = get_candidate_features(model_df)

    def _clean(part):
        return (
            part.replace([np.inf, -np.inf], np.nan)
                .dropna(subset=features + [TARGET_COLUMN])
                .copy()
        )

    train_ready = _clean(model_df.iloc[:split_idx])
    test_ready = _clean(model_df.iloc[split_idx:])
    assert train_ready["date"].max() < test_ready["date"].min()
    return train_ready, test_ready, features
