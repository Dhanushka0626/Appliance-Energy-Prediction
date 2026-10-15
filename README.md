# Appliance Energy Prediction

Deep-learning forecasting of household appliance energy consumption. The task is **one-step-ahead forecasting**:
predict the energy used by the appliances of a house (in Wh) in the **next 10-minute interval**,
using only information that is available before that interval starts.

The project covers the full workflow: exploratory data analysis, preprocessing, feature engineering and selection,
model design (LSTM, GRU, CNN-LSTM, MLP) with baseline models, training and evaluation, and model optimization
(hyper-parameter search, dropout, early stopping). A written report is included in [report/](report/).

## Results

Test set = the last 20 % of the time series (3,947 intervals, 30 Apr – 27 May 2016). All errors are in Wh.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Persistence (previous interval) | 26.65 | 66.44 | 0.467 |
| Linear Regression | 27.13 | 59.82 | 0.568 |
| Random Forest | 26.33 | 58.40 | 0.588 |
| MLP, 1 hidden layer (mean of 3 seeds) | 27.38 | 57.95 | 0.595 |
| LSTM, 1 layer (mean of 3 seeds) | 25.78 | 57.23 | 0.605 |
| **LSTM, 1 layer, saved model `models/trained_model.h5` (seed 42)** | **25.43** | **56.96** | **0.608** |
| LSTM, ensemble of 3 seeds | 25.45 | 56.91 | 0.609 |

* The optimized LSTM is clearly better than persistence and linear regression, and (as a 3-seed ensemble) better than the
  random forest by about 1.5 Wh RMSE (95 % interval −2.7 to −0.4 from a block bootstrap over test days).
* Differences among the deep-learning models and their tuning are small (about 1–2 %) and often within the random-seed
  variation; the optimization mostly confirmed that the starting LSTM was already near a flat optimum, and halved the
  size of the MLP (11 k parameters) at the same accuracy. The report discusses this in detail.
* The saved model is a single network (seed 42). The numbers above come from the notebook (section 5) and from
  `python src/train.py`, which reproduces the seed-42 result exactly.

## Repository structure

```
├── data/
│   ├── raw/
│   │   └── energy_data_set.csv          original data (public data set, see Data)
│   └── processed/
│       └── model_data.csv               date, target, the 14 selected features and the split label (train / validation / test)
├── notebooks/
│   └── Appliance_Energy_Prediction.ipynb   complete analysis: EDA, preprocessing, features, models, optimization
├── src/
│   ├── data_preprocessing.py            loading, cleaning, outliers, chronological split, scaling, sequence windows, metrics
│   ├── feature_engineering.py           leakage-free features, feature selection (voting + walk-forward test)
│   ├── model.py                         LSTM / GRU / CNN-LSTM / MLP builder, final configurations, save and load
│   └── train.py                         trains the final model from the raw data end to end
├── models/
│   ├── trained_model.h5                 final single-layer LSTM (22,337 parameters)
│   ├── scalers.joblib                   input and target scalers, feature list and look-back (needed for inference)
│   └── model_config.json                configuration, features and metrics of the saved model
├── report/
│   └── Appliance_Energy_Prediction_Report.pdf
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Usage

**Train the final model from the raw data** (about one minute on a CPU):

```bash
python src/train.py                                  # LSTM, seed 42 -> models/trained_model.h5
python src/train.py --model mlp --output trained_model_mlp.h5
python src/train.py --reselect                       # repeat the feature selection first (a few minutes)
```

The script cleans the data, builds the features, splits the data chronologically (first 80 % training, last 20 %
test; the validation set is the last 15 % of the training part), fits the scalers on the training sub-set only, trains
with early stopping, prints validation and test metrics, and writes the model, the scalers and `data/processed/model_data.csv`.

**Use the saved model** (predicts the consumption of the interval of a row of the processed table):

```python
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
import model as md

model, sc = md.load_bundle("models")                          # network + scalers + feature list
table = pd.read_csv("data/processed/model_data.csv", parse_dates=["date"])

X = sc["x_scaler"].transform(table[sc["features"]]).astype(np.float32)
i, L = len(table) - 1, sc["lookback"]                         # last row; window = rows i-L+1 ... i
window = X[i - L + 1: i + 1][None]                            # shape (1, look-back, features)
print(md.predict_wh(model, window, sc["y_scaler"]))           # forecast in Wh
```

**Notebook:** open `notebooks/Appliance_Energy_Prediction.ipynb` in Jupyter or Google Colab. On Colab it reads the data
from Google Drive; locally it reads `data/raw/energy_data_set.csv` (or the file named in the `ENERGY_CSV` environment variable).
The notebook writes its result tables and a copy of the model to a `results/` folder next to it.

## Method in brief

* **Leakage-free setup.** Features of an interval use only earlier consumption (lags, rolling statistics), sensor readings of
  the *previous* interval, and the calendar of the predicted interval; an automatic test verifies this.
* **Splits.** Chronological 80 % / 20 % train / test (never shuffled). A validation set is taken from the training part only;
  the test set is used once, for the final comparison.
* **Features.** Lags 10 min – 24 h (chosen from the autocorrelation), rolling mean / std, time of day. Raw indoor/outdoor sensors
  were removed by a walk-forward test because they drift with the season (mean outdoor temperature ≈ 6 °C in the training weeks
  and ≈ 14 °C in the test weeks).
* **Optimization.** Random search and grid search, evaluated with walk-forward cross-validation inside the training part and
  several seeds; dropout and early-stopping sweeps; a paired block bootstrap to judge the final differences.

## Data

`data/raw/energy_data_set.csv` is a public data set of appliance energy consumption together with indoor and outdoor
measurements (19,735 observations at 10-minute intervals, 11 Jan – 27 May 2016, 29 columns). It was obtained from the
UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction>.

## Limitations

* Only about 4.5 months of data from a single site, split 80 % / 20 %: other seasons and sites are not covered.
* The target is spiky; bursts above 400 Wh are not anticipated by any model, so the remaining error is dominated by them.
* The saved model is a single network; averaging several seeds lowers the error slightly at three times the inference cost.

## License

MIT License, see [LICENSE](LICENSE).
