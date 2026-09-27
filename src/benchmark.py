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
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.baselines import CNNLSTMModel, LSTMModel
from src.data_prep import fetch_macro_data, prepare_train_validation_test
from src.fitness_function import compute_directional_accuracy
from src.model import CNN_LSTM
from src.training import train_with_early_stopping


SEEDS = (42, 123, 2026, 7, 99)
TICKER = "AAPL"
START_DATE = "2015-01-01"
MAX_EPOCHS = 120
PATIENCE = 15

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


def set_seed(seed=SEEDS[0]):
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


def evaluate_model(model, X_test, y_test, scaler_y, keep_last=None):
    """Evaluate on a common suffix of the held-out test partition.

    Different lookback windows create different numbers of sequences.  Using
    the last N test sequences for every model aligns all metrics to the same
    chronological target observations.
    """
    if keep_last is not None:
        X_test = X_test[-keep_last:]
        y_test = y_test[-keep_last:]

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
        "test_samples": int(len(y_true)),
    }


def prepare_dataset(df_raw, window_size):
    data = prepare_train_validation_test(
        df_raw,
        window_size=window_size,
        save_scalers=False,
    )
    X_train, y_train, X_val, y_val, X_test, y_test, _, scaler_y = data
    if X_train is None:
        raise RuntimeError("Dataset is too short after preprocessing.")
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler_y


def run_one(
    name, model_factory, df_raw, config, seed, optimized=False,
    common_test_samples=None,
):
    print(f"\n{'=' * 68}\n{name}\n{'=' * 68}")
    set_seed(seed)

    X_train, y_train, X_val, y_val, X_test, y_test, scaler_y = prepare_dataset(
        df_raw,
        config["window_size"],
    )

    model = model_factory(X_train.shape[2], config)
    training = train_with_early_stopping(
        model, X_train, y_train, X_val, y_val,
        config["learning_rate"], config["batch_size"], DEVICE,
        max_epochs=MAX_EPOCHS, patience=PATIENCE,
    )
    metrics = evaluate_model(
        training.model, X_test, y_test, scaler_y,
        keep_last=common_test_samples,
    )

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ticker": TICKER,
        "model": name,
        **metrics,
        "optimized_by_ga_woa": optimized,
        "max_epochs": MAX_EPOCHS,
        "epochs_ran": training.epochs_ran,
        "best_epoch": training.best_epoch,
        "seed": seed,
        "hidden_layer_size": config["hidden_layer_size"],
        "dropout_rate": config["dropout_rate"],
        "cnn_filters": config.get("cnn_filters", ""),
        "num_layers": config["num_layers"],
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "window_size": config["window_size"],
    }


def main():
    set_seed(SEEDS[0])
    df_raw = load_raw_data()

    rows = []

    model_specs = [
        (
            "LSTM",
            lambda input_size, c: LSTMModel(
                input_size=input_size,
                hidden_layer_size=c["hidden_layer_size"],
                dropout_rate=c["dropout_rate"],
                num_layers=c["num_layers"],
            ),
            BASELINE_CONFIG.copy(),
            False,
        ),
        (
            "CNN-LSTM",
            lambda input_size, c: CNNLSTMModel(
                input_size=input_size,
                hidden_layer_size=c["hidden_layer_size"],
                dropout_rate=c["dropout_rate"],
                cnn_filters=c["cnn_filters"],
                num_layers=c["num_layers"],
            ),
            BASELINE_CONFIG.copy(),
            False,
        ),
        (
            "CNN-LSTM-Attention",
            lambda input_size, c: CNN_LSTM(
                input_size=input_size,
                hidden_layer_size=c["hidden_layer_size"],
                dropout_rate=c["dropout_rate"],
                cnn_filters=c["cnn_filters"],
                num_layers=c["num_layers"],
            ),
            BASELINE_CONFIG.copy(),
            False,
        ),
    ]

    with open("artifacts/model_config.json", "r", encoding="utf-8") as f:
        ga_config_raw = json.load(f)

    ga_config = {
        "hidden_layer_size": ga_config_raw["hidden_layer_size"],
        "dropout_rate": ga_config_raw["dropout_rate"],
        "cnn_filters": ga_config_raw["cnn_filters"],
        "num_layers": ga_config_raw["num_layers"],
        "window_size": ga_config_raw["window_size"],
        # New training runs persist these values. Older artifacts fall back
        # to shared defaults so the benchmark remains runnable.
        "learning_rate": ga_config_raw.get(
            "learning_rate",
            BASELINE_CONFIG["learning_rate"],
        ),
        "batch_size": ga_config_raw.get(
            "batch_size",
            BASELINE_CONFIG["batch_size"],
        ),
    }

    model_specs.append(
        (
            "GA-WOA CNN-LSTM-Attention",
            lambda input_size, c: CNN_LSTM(
                input_size=input_size,
                hidden_layer_size=c["hidden_layer_size"],
                dropout_rate=c["dropout_rate"],
                cnn_filters=c["cnn_filters"],
                num_layers=c["num_layers"],
            ),
            ga_config,
            True,
        )
    )

    # Fair-comparison boundary: each lookback can produce a slightly different
    # number of sequences.  Align evaluation to the shortest held-out test
    # suffix so every model is scored on identical final target observations.
    test_lengths = {}
    for name, _, config, _ in model_specs:
        *_, X_test, y_test, _ = prepare_dataset(df_raw, config["window_size"])
        test_lengths[name] = len(y_test)
    common_test_samples = min(test_lengths.values())
    print(f"Common aligned test samples: {common_test_samples}")
    print(f"Raw test lengths by model: {test_lengths}")

    for seed in SEEDS:
        for name, factory, config, optimized in model_specs:
            rows.append(run_one(
                name, factory, df_raw, config, seed, optimized,
                common_test_samples=common_test_samples,
            ))

    results = pd.DataFrame(rows)
    os.makedirs("experiments", exist_ok=True)
    output_path = "experiments/benchmark_results.csv"
    results.to_csv(output_path, index=False)

    metrics = ["rmse_return", "mae_return", "directional_accuracy"]
    summary = results.groupby("model", sort=False)[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv("experiments/benchmark_summary.csv", index=False)
    _plot_metric_comparisons(summary)

    print("\nBenchmark complete")
    print(results[[
        "model",
        "rmse_return",
        "mae_return",
        "directional_accuracy",
    ]].to_string(index=False))
    print(f"\nAll models evaluated on the same final {common_test_samples} test observations.")
    print(f"Saved: {output_path}")


def _plot_metric_comparisons(summary):
    import matplotlib.pyplot as plt

    os.makedirs("experiments/plots", exist_ok=True)
    for metric, filename, label in (
        ("rmse_return", "model_rmse_comparison.png", "RMSE"),
        ("mae_return", "model_mae_comparison.png", "MAE"),
        ("directional_accuracy", "model_directional_accuracy.png", "Directional accuracy"),
    ):
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.bar(
            summary["model"], summary[f"{metric}_mean"],
            yerr=summary[f"{metric}_std"], capsize=4,
        )
        axis.set_ylabel(f"{label} (mean ± std)")
        axis.tick_params(axis="x", rotation=20)
        figure.tight_layout()
        figure.savefig(os.path.join("experiments/plots", filename), dpi=150)
        plt.close(figure)


if __name__ == "__main__":
    main()
