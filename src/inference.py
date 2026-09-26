import json
import logging

import joblib
import numpy as np
import pandas as pd
import torch
import yfinance as yf

from src.data_prep import FEATURE_COLS, add_technical_indicators, compute_cmf
from src.model import CNN_LSTM

logger = logging.getLogger("GenLSTM_Production")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_system(
    config_path="./src/model_config.json",
    model_path="./src/best_model.pth",
    scaler_x_path="./src/scaler_x.pkl",
    scaler_y_path="./src/scaler_y.pkl",
):
    """Load trained model, scalers and window size for inference."""
    try:
        logger.info("Đang khởi tạo cấu trúc hệ thống và nạp trọng số mô hình...")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        model = CNN_LSTM(
            input_size=config["input_size"],
            hidden_layer_size=config["hidden_layer_size"],
            dropout_rate=config["dropout_rate"],
            cnn_filters=config["cnn_filters"],
            num_layers=config["num_layers"],
        )
        model.load_state_dict(
            torch.load(model_path, map_location=DEVICE, weights_only=True)
        )
        model.to(DEVICE)
        model.eval()

        scaler_x = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)
        window_size = config["window_size"]

        logger.info(
            "Khởi động thành công! Thiết bị: %s | Window Size: %s",
            DEVICE.type.upper(),
            window_size,
        )
        return model, scaler_x, scaler_y, window_size

    except Exception as exc:
        logger.error(
            "Lỗi nghiêm trọng khi khởi động hệ thống: %s",
            exc,
            exc_info=True,
        )
        return None, None, None, 16


def build_features_raw(ticker: str):
    """Build inference features using the same feature pipeline as training."""
    stock = yf.download(ticker, period="130d", progress=False)
    if stock.empty:
        return None, None

    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = stock.columns.get_level_values(0)

    stock["Volume"] = np.log1p(stock["Volume"])
    stock = add_technical_indicators(stock)

    try:
        macro_data = yf.download(["^VIX", "^TNX"], period="130d", progress=False)

        if isinstance(macro_data.columns, pd.MultiIndex):
            macro_data.columns = [
                f"{col[0]}_{col[1]}" for col in macro_data.columns
            ]

        stock["VIX"] = macro_data.get("Close_^VIX")
        stock["TNX"] = macro_data.get("Close_^TNX")

    except Exception as exc:
        logger.warning(
            "Không tải được dữ liệu vĩ mô thời gian thực, dùng fallback. Chi tiết: %s",
            exc,
        )
        stock["VIX"] = 20.0
        stock["TNX"] = 4.0

    stock["Sentiment_Score"] = compute_cmf(stock, period=20)

    stock[["VIX", "TNX"]] = stock[["VIX", "TNX"]].ffill().bfill()
    stock.dropna(subset=FEATURE_COLS, inplace=True)

    if stock.empty:
        return None, None

    return stock, float(stock["Close"].iloc[-1])


def predict_with_uncertainty(
    model,
    scaler_y,
    input_tensor,
    n_samples: int = 100,
    confidence: float = 0.90,
):
    """Run Monte Carlo Dropout inference and return prediction distribution."""
    model.train()

    raw_preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred_scaled = model(input_tensor).cpu().numpy()
            pred_return = float(
                scaler_y.inverse_transform(pred_scaled)[0][0]
            )
            raw_preds.append(pred_return)

    model.eval()

    preds = np.asarray(raw_preds, dtype=float)
    alpha = (1 - confidence) / 2

    return {
        "mean_return": float(np.mean(preds)),
        "std_return": float(np.std(preds)),
        "lower_return": float(np.percentile(preds, alpha * 100)),
        "upper_return": float(np.percentile(preds, (1 - alpha) * 100)),
        "n_samples": n_samples,
        "confidence": confidence,
        "all_samples": preds.tolist(),
    }
