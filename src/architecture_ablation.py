"""Controlled Phase-1 architecture ablation.

All variants use the SAME saved GA-WOA hyperparameters, fixed raw target
boundaries, training procedure, five seeds, and held-out test dates. Only the
network architecture changes. This isolates the contribution of CNN and
attention more cleanly than the main benchmark, whose fixed baselines use a
different shared configuration.

Run:
    python -m src.architecture_ablation
"""

import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.baselines import CNNLSTMModel, LSTMModel
from src.data_prep import fetch_macro_data, prepare_fixed_boundary_dataset
from src.fitness_function import compute_directional_accuracy
from src.model import CNN_LSTM
from src.training import train_with_early_stopping


SEEDS = (42, 123, 2026, 7, 99)
TICKER = "AAPL"
START_DATE = "2015-01-01"
MAX_EPOCHS = 120
PATIENCE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_raw_data():
    df = yf.download(TICKER, start=START_DATE, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        raise RuntimeError(f"No data returned for {TICKER}")

    macro = fetch_macro_data(
        TICKER,
        df.index[0].strftime("%Y-%m-%d"),
        df.index[-1].strftime("%Y-%m-%d"),
        df.index,
    )
    df["VIX"] = macro["VIX"]
    df["TNX"] = macro["TNX"]
    return df


def load_config():
    with open("artifacts/model_config.json", "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    required = (
        "hidden_layer_size", "dropout_rate", "cnn_filters", "num_layers",
        "window_size", "learning_rate", "batch_size",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise RuntimeError(f"model_config.json is missing: {missing}")
    return {key: raw[key] for key in required}


def evaluate(model, X_test, y_test, scaler_y):
    model.eval()
    with torch.no_grad():
        pred_scaled = model(
            torch.as_tensor(X_test, dtype=torch.float32, device=DEVICE)
        ).cpu().numpy().reshape(-1, 1)

    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred = scaler_y.inverse_transform(pred_scaled).flatten()
    return {
        "rmse_return": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_return": float(mean_absolute_error(y_true, y_pred)),
        "directional_accuracy": float(
            compute_directional_accuracy(y_true, y_pred)
        ),
    }


def main():
    config = load_config()
    df_raw = load_raw_data()
    data = prepare_fixed_boundary_dataset(
        df_raw, window_size=int(config["window_size"]),
    )
    (
        X_train, y_train, X_val, y_val, X_test, y_test,
        _, scaler_y, val_dates, test_dates,
    ) = data

    factories = (
        (
            "LSTM",
            lambda: LSTMModel(
                X_train.shape[2],
                int(config["hidden_layer_size"]),
                float(config["dropout_rate"]),
                int(config["num_layers"]),
            ),
        ),
        (
            "CNN-LSTM",
            lambda: CNNLSTMModel(
                X_train.shape[2],
                int(config["hidden_layer_size"]),
                float(config["dropout_rate"]),
                int(config["cnn_filters"]),
                int(config["num_layers"]),
            ),
        ),
        (
            "CNN-LSTM-Attention",
            lambda: CNN_LSTM(
                X_train.shape[2],
                int(config["hidden_layer_size"]),
                float(config["dropout_rate"]),
                int(config["cnn_filters"]),
                int(config["num_layers"]),
            ),
        ),
    )

    rows = []
    for seed in SEEDS:
        for name, factory in factories:
            set_seed(seed)
            result = train_with_early_stopping(
                factory(),
                X_train, y_train, X_val, y_val,
                float(config["learning_rate"]),
                int(config["batch_size"]),
                DEVICE,
                max_epochs=MAX_EPOCHS,
                patience=PATIENCE,
            )
            metrics = evaluate(result.model, X_test, y_test, scaler_y)
            rows.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "ticker": TICKER,
                "model": name,
                **metrics,
                "seed": seed,
                "test_samples": len(y_test),
                "validation_start": str(val_dates[0])[:10],
                "validation_end": str(val_dates[-1])[:10],
                "test_start": str(test_dates[0])[:10],
                "test_end": str(test_dates[-1])[:10],
                "best_epoch": result.best_epoch,
                "epochs_ran": result.epochs_ran,
                **config,
            })

    results = pd.DataFrame(rows)
    os.makedirs("experiments", exist_ok=True)
    results.to_csv("experiments/architecture_ablation.csv", index=False)

    metrics = ["rmse_return", "mae_return", "directional_accuracy"]
    summary = results.groupby("model", sort=False)[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv("experiments/architecture_ablation_summary.csv", index=False)

    print("\nControlled architecture ablation complete")
    print(f"All variants use {len(y_test)} identical held-out target observations.")
    print(summary.to_string(index=False))
    print("Saved: experiments/architecture_ablation.csv")
    print("Saved: experiments/architecture_ablation_summary.csv")


if __name__ == "__main__":
    main()
