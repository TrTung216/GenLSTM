"""Benchmark baseline models against the GA-WOA optimized model.

All models use the same raw dataset, preprocessing pipeline, temporal train/test split,
loss function, optimizer family and evaluation metrics. Baselines use one shared fixed
hyperparameter configuration. The optimized model uses the saved GA-WOA configuration.
"""

import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from src.baselines import CNNLSTMModel, LSTMModel
from src.data_prep import fetch_macro_data, prepare_data_from_df
from src.fitness_function import compute_directional_accuracy
from src.model import CNN_LSTM


SEED = 42
TICKER = "AAPL"
START_DATE = "2015-01-01"
EPOCHS = 120

BASELINE_CONFIG = {
    "hidden_layer_size": 64,
    "dropout_rate": 0.10,
    "cnn_filters": 32,
    "num_layers": 1,
    "learning_rate": 0.001,
    "batch_size": 32,
    "window_size": 60,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_raw_data():
    print(f"Downloading {TICKER} data...")
    df = yf.download(TICKER, start=START_DATE, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        raise RuntimeError(f"No data returned for {TICKER}")

    start_str = df.index[0].strftime("%Y-%m-%d")
    end_str = df.index[-1].strftime("%Y-%m-%d")
    macro = fetch_macro_data(TICKER, start_str, end_str, df.index)
    df["VIX"] = macro["VIX"]
    df["TNX"] = macro["TNX"]
    return df


def train_model(model, X_train, y_train, learning_rate, batch_size):
    model = model.to(DEVICE)
    loss_fn = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for seq, labels in loader:
            seq = seq.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            loss = loss_fn(model(seq), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / max(len(loader), 1)
            print(f"  epoch {epoch + 1:3d}/{EPOCHS} | loss={avg_loss:.6f}")

    return model


def evaluate_model(model, X_test, y_test, scaler_y):
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        preds_scaled = model(X_test_t).cpu().numpy().reshape(-1, 1)

    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred = scaler_y.inverse_transform(preds_scaled).flatten()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    da = float(compute_directional_accuracy(y_true, y_pred))

    return {
        "rmse_return": rmse,
        "mae_return": mae,
        "directional_accuracy": da,
    }


def prepare_dataset(df_raw, window_size):
    X_train, y_train, X_test, y_test, scaler_y = prepare_data_from_df(
        df_raw,
        window_size=window_size,
        save_scalers=False,
    )
    if X_train is None:
        raise RuntimeError("Dataset is too short after preprocessing.")
    return X_train, y_train, X_test, y_test, scaler_y


def run_one(name, model_factory, df_raw, config, optimized=False):
    print(f"\n{'=' * 68}\n{name}\n{'=' * 68}")
    set_seed()

    X_train, y_train, X_test, y_test, scaler_y = prepare_dataset(
        df_raw,
        config["window_size"],
    )

    model = model_factory(X_train.shape[2], config)
    model = train_model(
        model,
        X_train,
        y_train,
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
    )
    metrics = evaluate_model(model, X_test, y_test, scaler_y)

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ticker": TICKER,
        "model": name,
        **metrics,
        "optimized_by_ga_woa": optimized,
        "epochs": EPOCHS,
        "seed": SEED,
        "hidden_layer_size": config["hidden_layer_size"],
        "dropout_rate": config["dropout_rate"],
        "cnn_filters": config.get("cnn_filters", ""),
        "num_layers": config["num_layers"],
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "window_size": config["window_size"],
    }


def main():
    set_seed()
    df_raw = load_raw_data()

    rows = []

    rows.append(
        run_one(
            "LSTM",
            lambda input_size, c: LSTMModel(
                input_size=input_size,
                hidden_layer_size=c["hidden_layer_size"],
                dropout_rate=c["dropout_rate"],
                num_layers=c["num_layers"],
            ),
            df_raw,
            BASELINE_CONFIG.copy(),
        )
    )

    rows.append(
        run_one(
            "CNN-LSTM",
            lambda input_size, c: CNNLSTMModel(
                input_size=input_size,
                hidden_layer_size=c["hidden_layer_size"],
                dropout_rate=c["dropout_rate"],
                cnn_filters=c["cnn_filters"],
                num_layers=c["num_layers"],
            ),
            df_raw,
            BASELINE_CONFIG.copy(),
        )
    )

    rows.append(
        run_one(
            "CNN-LSTM-Attention",
            lambda input_size, c: CNN_LSTM(
                input_size=input_size,
                hidden_layer_size=c["hidden_layer_size"],
                dropout_rate=c["dropout_rate"],
                cnn_filters=c["cnn_filters"],
                num_layers=c["num_layers"],
            ),
            df_raw,
            BASELINE_CONFIG.copy(),
        )
    )

    with open("artifacts/model_config.json", "r", encoding="utf-8") as f:
        ga_config_raw = json.load(f)

    ga_config = {
        "hidden_layer_size": ga_config_raw["hidden_layer_size"],
        "dropout_rate": ga_config_raw["dropout_rate"],
        "cnn_filters": ga_config_raw["cnn_filters"],
        "num_layers": ga_config_raw["num_layers"],
        "window_size": ga_config_raw["window_size"],
        # Current artifact does not persist optimizer LR/batch, so use the
        # shared training defaults here and report them explicitly.
        "learning_rate": BASELINE_CONFIG["learning_rate"],
        "batch_size": BASELINE_CONFIG["batch_size"],
    }

    rows.append(
        run_one(
            "GA-WOA CNN-LSTM-Attention",
            lambda input_size, c: CNN_LSTM(
                input_size=input_size,
                hidden_layer_size=c["hidden_layer_size"],
                dropout_rate=c["dropout_rate"],
                cnn_filters=c["cnn_filters"],
                num_layers=c["num_layers"],
            ),
            df_raw,
            ga_config,
            optimized=True,
        )
    )

    results = pd.DataFrame(rows)
    os.makedirs("experiments", exist_ok=True)
    output_path = "experiments/benchmark_results.csv"
    results.to_csv(output_path, index=False)

    print("\nBenchmark complete")
    print(results[[
        "model",
        "rmse_return",
        "mae_return",
        "directional_accuracy",
    ]].to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
