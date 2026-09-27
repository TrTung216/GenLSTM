"""Run one GA-WOA fitness ablation profile without overwriting production artifacts.

Examples:
    python -m src.fitness_ablation --profile A
    python -m src.fitness_ablation --profile B
    python -m src.fitness_ablation --profile C

Profiles:
    A = 50% RMSE + 50% MAE
    B = 50% RMSE + 50% Directional Accuracy
    C = 40% RMSE + 40% Directional Accuracy + 20% Drawdown

Each run performs a fresh GA-WOA search with walk-forward validation, trains the
selected configuration once using the standard train/validation split, evaluates
only once on the held-out test partition, and appends the result to
experiments/fitness_ablation.csv. Production artifacts are not modified.
"""

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src import ga_lstm
from src.data_prep import fetch_macro_data, prepare_fixed_boundary_dataset
from src.fitness_function import (
    FITNESS_PROFILES,
    CNN_LSTM,
    compute_directional_accuracy,
)
from src.training import train_with_early_stopping


SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_raw_data():
    df_raw = yf.download(
        ga_lstm.TICKER_SYMBOL,
        start=ga_lstm.START_DATE,
        end=ga_lstm.END_DATE,
        progress=False,
    )
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    if df_raw.empty or len(df_raw) < 100:
        raise RuntimeError("No usable market data returned.")

    start_str = df_raw.index[0].strftime("%Y-%m-%d")
    end_str = df_raw.index[-1].strftime("%Y-%m-%d")
    macro = fetch_macro_data(
        ga_lstm.TICKER_SYMBOL,
        start_str,
        end_str,
        df_raw.index,
    )
    df_raw["VIX"] = macro["VIX"]
    df_raw["TNX"] = macro["TNX"]
    return df_raw


def evaluate_selected_config(df_raw, chromosome):
    units, dropout, lr, batch, window, filters, layers = chromosome
    package = prepare_fixed_boundary_dataset(
        df_raw,
        int(window),
        save_scalers=False,
    )
    (
        X_train, y_train, X_val, y_val, X_test, y_test,
        _, scaler_y, _, _,
    ) = package

    model = CNN_LSTM(
        input_size=X_train.shape[2],
        hidden_layer_size=int(units),
        dropout_rate=float(dropout),
        cnn_filters=int(filters),
        num_layers=int(layers),
    ).to(ga_lstm.device)

    result = train_with_early_stopping(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        float(lr),
        int(batch),
        ga_lstm.device,
        max_epochs=120,
        patience=15,
    )
    model = result.model
    model.eval()
    with torch.no_grad():
        pred_scaled = model(
            torch.as_tensor(X_test, dtype=torch.float32, device=ga_lstm.device)
        ).cpu().numpy().reshape(-1, 1)

    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred = scaler_y.inverse_transform(pred_scaled).flatten()
    return {
        "rmse_return": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_return": float(mean_absolute_error(y_true, y_pred)),
        "directional_accuracy": float(
            compute_directional_accuracy(y_true, y_pred)
        ),
        "best_epoch": int(result.best_epoch),
        "epochs_ran": int(result.epochs_ran),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=sorted(FITNESS_PROFILES),
        required=True,
        help="Fitness ablation profile: A, B, or C.",
    )
    args = parser.parse_args()
    profile = args.profile
    weights = FITNESS_PROFILES[profile].copy()

    set_seed()
    print(f"Fitness ablation profile {profile}: {weights}")
    df_raw = load_raw_data()

    # run_ga_lstm reads this module-level config.  Changing it here affects only
    # this process and does not alter source files or production artifacts.
    ga_lstm.FITNESS_WEIGHTS = weights
    best, history, components, stats = ga_lstm.run_ga_lstm(df_raw)

    os.makedirs("experiments", exist_ok=True)
    pd.DataFrame(stats).to_csv(
        f"experiments/ga_generation_stats_profile_{profile}.csv",
        index=False,
    )

    set_seed()
    metrics = evaluate_selected_config(df_raw, best)

    units, dropout, lr, batch, window, filters, layers = best
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "profile": profile,
        "fitness_weights": json.dumps(weights, sort_keys=True),
        **metrics,
        "units": int(units),
        "dropout": float(dropout),
        "learning_rate": float(lr),
        "batch_size": int(batch),
        "window_size": int(window),
        "cnn_filters": int(filters),
        "num_layers": int(layers),
        "search_best_fitness": float(history[-1]),
        "seed": SEED,
    }

    path = "experiments/fitness_ablation.csv"
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not os.path.exists(path),
        index=False,
    )

    print("\nFitness ablation complete")
    print(pd.DataFrame([row])[[
        "profile",
        "rmse_return",
        "mae_return",
        "directional_accuracy",
        "search_best_fitness",
    ]].to_string(index=False))
    print(f"Saved: {path}")
    print("Production artifacts were not modified.")


if __name__ == "__main__":
    main()
