"""Feature engineering and feature selection for the Appliance Energy Prediction project.

Mirrors section 2 of the notebook. The model predicts the NEXT 10-minute
interval, so the features of row i are built only from information available
before interval i:

* lags / rolling statistics use earlier `Appliances` values only,
* sensor and weather readings come from the previous interval (shifted by one),
* calendar features describe interval i itself (known in advance).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "Appliances"
SEED = 42

# Lags chosen from the autocorrelation analysis (10, 20, 30 min, 1, 2, 3 h, 24 h)
LAGS = [1, 2, 3, 6, 12, 18, 144]

# Public holidays inside the measurement period (11 Jan - 27 May 2016):
# Easter Monday, Labour Day, Ascension, Whit Monday
HOLIDAYS = pd.to_datetime(["2016-03-28", "2016-05-01", "2016-05-05", "2016-05-16"]).date

INDOOR_T = ["T1", "T2", "T3", "T4", "T5", "T7", "T8", "T9"]       # T6 is left out: it follows the outdoor temperature (mean 7.9 °C, min -6 °C)
INDOOR_RH = ["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_7", "RH_8", "RH_9"]
EXCLUDED = {"date", TARGET_COLUMN, "hour", "day_of_week", "month"}  # sin/cos versions are used instead

# The 14 features chosen by the selection in the notebook (voting + walk-forward feature-group test),
# in the order produced there: past consumption (lags, rolling statistics) and the time of day.
SELECTED_FEATURES = [
    "appliances_lag_1", "appliances_rolling_mean_1h", "appliances_lag_2", "appliances_rolling_std_3h",
    "appliances_lag_3", "hour_sin", "appliances_rolling_mean_3h", "hour_cos", "appliances_lag_6",
    "appliances_lag_18", "appliances_rolling_std_1h", "appliances_lag_12", "appliances_rolling_mean_24h",
    "appliances_lag_144",
]

OUTDOOR_COLS = ["T6", "RH_6", "T_out", "RH_out", "Press_mm_hg", "Windspeed", "Visibility",
                "Tdewpoint", "Tout_RHout_interaction", "indoor_outdoor_temp_diff"]
TIME_COLS = ["hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
             "is_weekend", "is_holiday"]


# ---------------------------------------------------------------------------
# Feature construction (notebook sections 2.1 - 2.5)
# ---------------------------------------------------------------------------
def build_features(raw):
    """Create all model features. Row i is built only from information known before interval i."""
    d = raw.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = (d.dropna(subset=["date"]).sort_values("date")
           .drop_duplicates(subset="date").reset_index(drop=True))

    # Sensor and weather readings come from the PREVIOUS interval (no look-ahead)
    sensors = [c for c in d.columns if c not in ("date", TARGET_COLUMN)]
    d[sensors] = d[sensors].shift(1)

    # 2.1 Time-based features of the interval being predicted
    d["hour"] = d["date"].dt.hour
    d["day_of_week"] = d["date"].dt.dayofweek
    d["month"] = d["date"].dt.month
    d["is_weekend"] = (d["day_of_week"] >= 5).astype(int)
    day_fraction = (d["date"].dt.hour * 60 + d["date"].dt.minute) / 1440
    d["hour_sin"] = np.sin(2 * np.pi * day_fraction)
    d["hour_cos"] = np.cos(2 * np.pi * day_fraction)
    d["day_sin"] = np.sin(2 * np.pi * d["day_of_week"] / 7)
    d["day_cos"] = np.cos(2 * np.pi * d["day_of_week"] / 7)
    d["month_sin"] = np.sin(2 * np.pi * (d["month"] - 1) / 12)
    d["month_cos"] = np.cos(2 * np.pi * (d["month"] - 1) / 12)

    # 2.2 Rolling statistics of PAST consumption (shift(1) excludes the current target)
    past = d[TARGET_COLUMN].shift(1)
    d["appliances_rolling_mean_1h"] = past.rolling(6, min_periods=6).mean()
    d["appliances_rolling_mean_3h"] = past.rolling(18, min_periods=18).mean()
    d["appliances_rolling_mean_24h"] = past.rolling(144, min_periods=144).mean()
    d["appliances_rolling_std_1h"] = past.rolling(6, min_periods=6).std()
    d["appliances_rolling_std_3h"] = past.rolling(18, min_periods=18).std()

    # 2.3 Lagged consumption
    for lag in LAGS:
        d[f"appliances_lag_{lag}"] = d[TARGET_COLUMN].shift(lag)

    # 2.4 Interaction features (computed from the already shifted sensor values)
    d["T2_RH2_interaction"] = d["T2"] * d["RH_2"]
    d["Tout_RHout_interaction"] = d["T_out"] * d["RH_out"]

    # 2.5 Domain-specific features
    d["is_holiday"] = d["date"].dt.date.isin(HOLIDAYS).astype(int)
    d["indoor_temp_mean"] = d[INDOOR_T].mean(axis=1)
    d["indoor_rh_mean"] = d[INDOOR_RH].mean(axis=1)
    d["indoor_outdoor_temp_diff"] = d["indoor_temp_mean"] - d["T_out"]
    d["lights_on"] = (d["lights"] > 0).astype(int)

    return d.replace([np.inf, -np.inf], np.nan)


def get_feature_pool(model_df):
    """Numeric candidate feature columns (target, date and raw integer calendar columns excluded)."""
    return [c for c in model_df.columns
            if c not in EXCLUDED and pd.api.types.is_numeric_dtype(model_df[c])]


def check_no_leakage(raw, row=5000):
    """Leakage test: changing ANY raw value of interval `row` must not change the features of that row."""
    perturbed = raw.copy()
    value_cols = [c for c in raw.columns if c != "date"]
    perturbed.loc[row, value_cols] = perturbed.loc[row, value_cols] + 10_000.0

    f_orig, f_pert = build_features(raw), build_features(perturbed)
    pool = get_feature_pool(f_orig)
    same = np.allclose(f_orig.loc[row, pool].astype(float),
                       f_pert.loc[row, pool].astype(float), equal_nan=True)
    next_changed = not np.allclose(f_orig.loc[row + 1, pool].astype(float),
                                   f_pert.loc[row + 1, pool].astype(float), equal_nan=True)
    assert same, "LEAKAGE: features of row i depend on interval i"
    assert next_changed, "sanity check failed: row i+1 should use interval i"
    return True


# ---------------------------------------------------------------------------
# Feature selection (notebook section 2.7) - uses training data only
# ---------------------------------------------------------------------------
def vote_features(X, y, min_votes=2, corr_threshold=0.05, rf_cumulative=0.95,
                  rfe_features=25, seed=SEED):
    """Select features by voting: correlation, random-forest importance and RFE (Ridge).

    A feature is kept when at least `min_votes` of the three methods choose it.
    Exactly duplicated columns (e.g. rv1 == rv2) are removed first.
    Returns (selected_features, votes_table).
    """
    duplicated = X.columns[X.T.duplicated(keep="first")].tolist()
    pool = [c for c in X.columns if c not in duplicated]

    corr = X[pool].corrwith(y).abs().sort_values(ascending=False)
    sel_corr = set(corr[corr >= corr_threshold].index)

    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.5,
                               random_state=seed, n_jobs=-1).fit(X[pool], y)
    importance = pd.Series(rf.feature_importances_, index=pool).sort_values(ascending=False)
    sel_rf = set(importance[importance.cumsum() - importance < rf_cumulative].index)

    rfe = RFE(Ridge(alpha=1.0), n_features_to_select=rfe_features, step=2)
    rfe.fit(StandardScaler().fit_transform(X[pool]), y)
    sel_rfe = set(np.array(pool)[rfe.support_])

    votes = pd.DataFrame({
        "corr |r|": corr.reindex(pool).round(3),
        "RF importance": importance.reindex(pool).round(4),
        "corr": [c in sel_corr for c in pool],
        "RF": [c in sel_rf for c in pool],
        "RFE": [c in sel_rfe for c in pool],
    }, index=pool)
    votes["votes"] = votes[["corr", "RF", "RFE"]].sum(axis=1)
    votes = votes.sort_values(["votes", "RF importance"], ascending=False)
    return votes.index[votes["votes"] >= min_votes].tolist(), votes


def candidate_feature_sets(voted_features, pool):
    """The candidate sets compared by the walk-forward group test."""
    past_cols = [c for c in pool if c.startswith("appliances_")]
    return {
        "voted set": voted_features,
        "voted, no environment columns": [c for c in voted_features if c in past_cols + TIME_COLS],
        "voted, no outdoor weather": [c for c in voted_features if c not in OUTDOOR_COLS],
        "all past-consumption + calendar": [c for c in pool if c in past_cols + TIME_COLS],
    }


def walk_forward_group_test(train_period, candidate_sets, n_splits=5, seed=SEED):
    """Compare feature sets with expanding-window validation inside the TRAINING period.

    Guards against raw sensor/weather columns that drift with the season.
    Returns (results_table, chosen_set_name): the lowest mean random-forest RMSE,
    preferring the smaller set among near-ties (within 1 %).
    """
    y = train_period[TARGET_COLUMN].to_numpy()
    folds = list(TimeSeriesSplit(n_splits=n_splits).split(train_period))

    rows = []
    for set_name, feats in candidate_sets.items():
        for model_name in ["Random Forest", "Ridge"]:
            rmse = []
            for tr, te in folds:
                if model_name == "Random Forest":
                    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=3,
                                                  max_features=0.5, random_state=seed, n_jobs=-1)
                else:
                    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                model.fit(train_period.iloc[tr][feats], y[tr])
                pred = model.predict(train_period.iloc[te][feats])
                rmse.append(np.sqrt(mean_squared_error(y[te], pred)))
            rows.append({"Feature set": set_name, "Model": model_name, "# features": len(feats),
                         "Mean RMSE (Wh)": np.mean(rmse), "Worst fold (Wh)": np.max(rmse)})
    results = pd.DataFrame(rows)

    rf = results[results["Model"] == "Random Forest"].set_index("Feature set")
    near_best = rf[rf["Mean RMSE (Wh)"] <= rf["Mean RMSE (Wh)"].min() * 1.01].sort_values("# features")
    return results, near_best.index[0]
