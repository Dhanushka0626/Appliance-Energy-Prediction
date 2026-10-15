"""Model definitions for the Appliance Energy Prediction project.

Mirrors sections 3 and 5 of the notebook: the network builder (LSTM, GRU, CNN-LSTM, MLP), the final
optimized configurations of the two light-weight models, training with early stopping, prediction in Wh,
and loading of a saved model together with its scalers.
"""

import json
import os

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, GRU, Input, LSTM

SEED = 42

# Final optimized configurations (notebook section 5). Both are single-layer models.
FINAL_CONFIGS = {
    "lstm": dict(arch="LSTM", units=(64,), dense=32, rnn_act="tanh", dense_act="relu", dropout=0.2, l2=0.0,
                 optimizer="adam", lr=1e-3, loss="mse", lookback=12, batch=128, patience=3, max_epochs=100),
    "mlp": dict(arch="MLP", units=(128,), dense=0, rnn_act="tanh", dense_act="relu", dropout=0.1, l2=0.0,
                optimizer="adam", lr=1e-3, loss="mse", lookback=6, batch=256, patience=6, max_epochs=100),
}


def make_optimizer(name, lr):
    """Adam, RMSprop, Nadam or SGD with momentum; gradient clipping (norm 1.0) keeps recurrent training stable."""
    name = name.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr, clipnorm=1.0)
    if name == "nadam":
        return tf.keras.optimizers.Nadam(learning_rate=lr, clipnorm=1.0)
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, clipnorm=1.0)
    raise ValueError(f"unknown optimizer: {name}")


def make_loss(name):
    """'mse', 'mae' or 'huber' (delta = 1 standard deviation of the standardized target)."""
    return tf.keras.losses.Huber(delta=1.0) if name == "huber" else name


def build_model(cfg, n_features):
    """Build and compile a network. `dense = 0` means no extra dense head (single-hidden-layer models)."""
    reg = regularizers.l2(cfg["l2"]) if cfg["l2"] else None
    layers = [Input(shape=(cfg["lookback"], n_features))]
    arch, units = cfg["arch"], cfg["units"]

    if arch in ("LSTM", "GRU"):
        rnn = LSTM if arch == "LSTM" else GRU
        for k, u in enumerate(units):
            layers += [rnn(u, activation=cfg["rnn_act"], return_sequences=k < len(units) - 1, kernel_regularizer=reg),
                       Dropout(cfg["dropout"])]
    elif arch == "CNN-LSTM":
        layers += [Conv1D(32, 3, padding="same", activation=cfg["dense_act"], kernel_regularizer=reg),
                   Conv1D(32, 3, padding="same", activation=cfg["dense_act"], kernel_regularizer=reg),
                   LSTM(units[0], activation=cfg["rnn_act"], kernel_regularizer=reg),
                   Dropout(cfg["dropout"])]
    elif arch == "MLP":
        layers.append(Flatten())
        for u in units:
            layers += [Dense(u, activation=cfg["dense_act"], kernel_regularizer=reg), Dropout(cfg["dropout"])]
    else:
        raise ValueError(f"unknown architecture: {arch}")

    if cfg["dense"]:
        layers.append(Dense(cfg["dense"], activation=cfg["dense_act"], kernel_regularizer=reg))
    layers.append(Dense(1))

    model = Sequential(layers)
    model.compile(optimizer=make_optimizer(cfg["optimizer"], cfg["lr"]), loss=make_loss(cfg["loss"]), metrics=["mae"])
    return model


def fit_model(cfg, X_train, y_train, X_val, y_val, seed=SEED, verbose=0):
    """Train with early stopping on the validation loss (best weights restored) and learning-rate reduction."""
    tf.keras.utils.set_random_seed(seed)
    model = build_model(cfg, X_train.shape[2])
    patience = cfg.get("patience", 6)
    callbacks = [EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
                 ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, patience // 2), min_lr=1e-5)]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=cfg.get("max_epochs", 100),
                        batch_size=cfg.get("batch", 128), callbacks=callbacks, verbose=verbose)
    return model, history.history


def predict_wh(model, X, y_scaler):
    """Predict and convert the standardized output back to Wh."""
    scaled = model.predict(X, batch_size=1024, verbose=0).reshape(-1, 1)
    return y_scaler.inverse_transform(scaled).ravel()


def save_bundle(model, x_scaler, y_scaler, features, cfg, metrics, models_dir, model_file="trained_model.h5", seed=SEED):
    """Save the network (.h5) together with everything needed to use it: scalers, feature list and settings."""
    os.makedirs(models_dir, exist_ok=True)
    model.save(os.path.join(models_dir, model_file))
    joblib.dump({"x_scaler": x_scaler, "y_scaler": y_scaler, "features": list(features), "lookback": cfg["lookback"]},
                os.path.join(models_dir, "scalers.joblib"))
    info = {"model_file": model_file, "seed": seed, "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
            "features": list(features), "metrics": metrics}
    with open(os.path.join(models_dir, "model_config.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)


def load_bundle(models_dir, model_file="trained_model.h5"):
    """Load a saved model and its scalers. Returns (model, scalers) where scalers holds x_scaler, y_scaler, features, lookback."""
    model = tf.keras.models.load_model(os.path.join(models_dir, model_file), compile=False)
    scalers = joblib.load(os.path.join(models_dir, "scalers.joblib"))
    return model, scalers
