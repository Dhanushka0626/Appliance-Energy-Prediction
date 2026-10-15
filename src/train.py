"""Train the final model of the Appliance Energy Prediction project from the raw data.

Usage (from the repository root):

    python src/train.py                              # final LSTM, seed 42
    python src/train.py --model mlp --output trained_model_mlp.h5
    python src/train.py --reselect                   # repeat the feature selection (a few minutes)

Pipeline (notebook sections 1-5): load and clean the raw data -> leakage-free features -> chronological
80 % / 20 % split with a validation set taken from the training part -> feature set (the 14 features chosen
in the notebook, or a new selection with --reselect) -> scaling fitted on the training sub-set -> sequence
windows -> training with early stopping -> evaluation in Wh -> saved model, scalers and processed data.
The test set is used for one final evaluation only.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # allow `python src/train.py`

import data_preprocessing as dp          # noqa: E402
import feature_engineering as fe         # noqa: E402
import model as md                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def select_features(full, pool, idx_sub, idx_val):
    """Repeat the feature selection of the notebook (voting + walk-forward feature-group test, training part only)."""
    voted, _ = fe.vote_features(full.loc[idx_sub, pool], full.loc[idx_sub, dp.TARGET_COLUMN])
    candidates = fe.candidate_feature_sets(voted, [c for c in pool if c != "rv2"])
    period = full.loc[np.concatenate([idx_sub, idx_val])]               # training part only - no test rows
    table, chosen = fe.walk_forward_group_test(period, candidates)
    print(table.round(2).to_string(index=False))
    print(f"-> selected feature set: '{chosen}' ({len(candidates[chosen])} features)")
    return candidates[chosen]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "raw", "energy_data_set.csv"), help="raw CSV file")
    ap.add_argument("--model", choices=sorted(md.FINAL_CONFIGS), default="lstm", help="model family (default: lstm)")
    ap.add_argument("--seed", type=int, default=md.SEED, help="random seed (default: 42)")
    ap.add_argument("--output", default="trained_model.h5", help="file name of the saved model inside models/")
    ap.add_argument("--models-dir", default=os.path.join(ROOT, "models"))
    ap.add_argument("--processed-dir", default=os.path.join(ROOT, "data", "processed"))
    ap.add_argument("--reselect", action="store_true", help="repeat the feature selection instead of using the saved list")
    args = ap.parse_args()

    # 1. load, clean, features, chronological split (80 % train / 20 % test; validation from the training part)
    raw = dp.clean_data(dp.load_data(args.data))
    quality = dp.check_data_quality(raw)
    print("data quality:", quality)
    fe.check_no_leakage(raw)
    print("leakage test passed")

    model_df = fe.build_features(raw)
    pool = fe.get_feature_pool(model_df)
    split_index = len(dp.chronological_split(raw)[0])
    full, idx_sub, idx_val, idx_test = dp.build_timeline(model_df, pool, split_index)
    print(f"rows: training sub-set {len(idx_sub)}, validation {len(idx_val)}, test {len(idx_test)}")

    # 2. feature set
    features = select_features(full, pool, idx_sub, idx_val) if args.reselect else list(fe.SELECTED_FEATURES)
    print(f"{len(features)} features:", features)

    # 3. processed data for inspection and re-use (date, target, selected features, split)
    split_label = np.empty(len(full), dtype=object)
    split_label[idx_sub], split_label[idx_val], split_label[idx_test] = "train", "validation", "test"
    processed = full[["date", dp.TARGET_COLUMN] + features].assign(split=split_label)
    os.makedirs(args.processed_dir, exist_ok=True)
    processed.to_csv(os.path.join(args.processed_dir, "model_data.csv"), index=False)

    # 4. scaling (fitted on the training sub-set only) and sequence windows
    cfg = md.FINAL_CONFIGS[args.model]
    x_scaler, y_scaler, X_scaled, y_scaled = dp.fit_scalers(full, features, idx_sub)
    windows = {name: dp.make_windows(X_scaled, y_scaled, idx, cfg["lookback"])
               for name, idx in [("train", idx_sub), ("val", idx_val), ("test", idx_test)]}
    (X_tr, y_tr, _), (X_va, y_va, _), (X_te, _, _) = windows["train"], windows["val"], windows["test"]
    y_val_wh = full.loc[idx_val, dp.TARGET_COLUMN].to_numpy()
    y_test_wh = full.loc[idx_test, dp.TARGET_COLUMN].to_numpy()

    # 5. training with early stopping on the validation loss
    print(f"\ntraining {cfg['arch']} (look-back {cfg['lookback']}, units {cfg['units']}, dropout {cfg['dropout']}, "
          f"batch {cfg['batch']}, patience {cfg['patience']}, seed {args.seed})")
    model, history = md.fit_model(cfg, X_tr, y_tr, X_va, y_va, seed=args.seed, verbose=0)
    print(f"stopped after {len(history['loss'])} epochs (best epoch {int(np.argmin(history['val_loss']) + 1)}), "
          f"{model.count_params():,} parameters")

    # 6. evaluation in Wh: validation for the design decisions, test for the one final report
    metrics = {"validation": dp.regression_metrics(y_val_wh, md.predict_wh(model, X_va, y_scaler)),
               "test": dp.regression_metrics(y_test_wh, md.predict_wh(model, X_te, y_scaler)),
               "persistence_test": dp.regression_metrics(y_test_wh, dp.persistence_forecast(full, idx_test))}
    for part, m in metrics.items():
        print(f"{part:17s} " + " | ".join(f"{k} {v:.3f}" for k, v in m.items()))

    # 7. save the model with its scalers and settings
    md.save_bundle(model, x_scaler, y_scaler, features, cfg, metrics, args.models_dir, model_file=args.output, seed=args.seed)
    print(f"\nsaved: {os.path.join(args.models_dir, args.output)} (+ scalers.joblib, model_config.json) and "
          f"{os.path.join(args.processed_dir, 'model_data.csv')}")


if __name__ == "__main__":
    main()
