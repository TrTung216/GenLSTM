import os
import time
import logging
from logging.handlers import RotatingFileHandler
import traceback

from flask import Flask, render_template, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from cachetools import TTLCache
import threading

import yfinance as yf
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import json

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# ⚙️ CẤU HÌNH LOGGING SẢN XUẤT (ROTATING LOGS)
# =========================================================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "prediction.log")

logger = logging.getLogger("GenLSTM_Production")
logger.setLevel(logging.INFO)

# Cấu hình file log: tối đa 5MB/file, lưu tối đa 5 file backup cũ
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5, encoding="utf-8")
file_formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Cấu hình console log để theo dõi thời gian thực (real-time khi chạy Docker/Terminal)
console_handler = logging.StreamHandler()
console_handler.setFormatter(file_formatter)
logger.addHandler(console_handler)


# ==========================================
# CACHE — lưu dữ liệu yfinance tránh gọi lại
# ==========================================
_stock_cache: TTLCache = TTLCache(maxsize=50, ttl=3600)
_cache_lock = threading.Lock()

def get_cached_stock(ticker: str):
    """
    Trả về (stock_df, last_close) từ cache nếu còn hạn,
    hoặc tải mới từ yfinance rồi lưu vào cache.
    """
    with _cache_lock:
        if ticker in _stock_cache:
            cached = _stock_cache[ticker]
            logger.info(f"[CACHE HIT] Ticker: {ticker} — Sử dụng dữ liệu lưu sẵn trong cache.")
            return cached["stock"], cached["last_close"], True

    # Cache miss → tải mới
    logger.info(f"[CACHE MISS] Ticker: {ticker} — Đang gửi yêu cầu tải mới từ API yfinance...")
    stock, last_close = build_features_raw(ticker)

    if stock is not None:
        with _cache_lock:
            _stock_cache[ticker] = {
                "stock"      : stock,
                "last_close" : last_close,
                "fetched_at" : time.time()
            }
    return stock, last_close, False


# ==========================================
# RATE LIMITER
# ==========================================
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

PREDICT_LIMITS = ["10 per minute", "100 per day"]


# ==========================================
# 1. ĐỊNH NGHĨA CẤU TRÚC AI
# ==========================================
class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size=50, dropout_rate=0.2, cnn_filters=16, num_layers=1):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            cnn_filters,
            hidden_layer_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.attention = nn.Linear(hidden_layer_size, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_layer_size, 1)

    def forward(self, input_seq):
        x = input_seq.permute(0, 2, 1)
        x = self.relu(self.conv1d(x))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        return self.linear(self.dropout(context_vector))


# ==========================================
# 2. HÀM LOAD HỆ THỐNG
# ==========================================
def load_system():
    try:
        logger.info("Đang khởi tạo cấu trúc hệ thống và nạp trọng số mô hình...")
        with open('./src/model_config.json', 'r') as f:
            config = json.load(f)

        net = CNN_LSTM(
            input_size=config['input_size'],
            hidden_layer_size=config['hidden_layer_size'],
            dropout_rate=config['dropout_rate'],
            cnn_filters=config['cnn_filters'],
            num_layers=config['num_layers']
        )
        net.load_state_dict(torch.load('./src/best_model.pth', map_location=device, weights_only=True))
        net.to(device)
        
        logger.info(f"✅ Khởi động thành công! Thiết bị: {device.type.upper()} | Window Size: {config['window_size']}")
        return net, joblib.load('./src/scaler_x.pkl'), joblib.load('./src/scaler_y.pkl'), config['window_size']

    except Exception as e:
        logger.error(f"❌ Lỗi nghiêm trọng khi khởi động hệ thống: {str(e)}", exc_info=True)
        return None, None, None, 16


model, scaler_x, scaler_y, WINDOW_SIZE = load_system()


# ==========================================
# 3. MC DROPOUT — HÀM DỰ BÁO CÓ KHOẢNG TIN CẬY
# ==========================================
def predict_with_uncertainty(input_tensor, n_samples: int = 100, confidence: float = 0.90):
    model.train()  # Bật train mode để giữ dropout hoạt động lúc inference

    raw_preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred_scaled = model(input_tensor).cpu().numpy()
            pred_return = float(scaler_y.inverse_transform(pred_scaled)[0][0])
            raw_preds.append(pred_return)

    model.eval()  # Trả về chế độ eval tiêu chuẩn

    preds = np.array(raw_preds)
    alpha = (1 - confidence) / 2

    return {
        "mean_return"  : float(np.mean(preds)),
        "std_return"   : float(np.std(preds)),
        "lower_return" : float(np.percentile(preds, alpha * 100)),
        "upper_return" : float(np.percentile(preds, (1 - alpha) * 100)),
        "n_samples"    : n_samples,
        "confidence"   : confidence,
        "all_samples"  : preds.tolist()
    }


# ==========================================
# 4. TIỀN XỬ LÝ DỮ LIỆU
# ==========================================
# Cập nhật danh sách biến toàn cục trong app.py để đồng nhất cấu trúc
FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume', 
    'SMA_10', 'SMA_20', 'EMA_20', 'RSI_14', 
    'MACD', 'Signal_Line', 'BB_Middle', 'BB_Upper', 'BB_Lower',
    'VIX', 'TNX', 'Sentiment_Score'
]

def build_features_raw(ticker: str):
    """Tải dữ liệu từ yfinance và tính chỉ báo kỹ thuật + Vĩ mô thực tế (Inference)."""
    # Tải dư ra 130 ngày để đảm bảo sau khi tính toán các đường MA(20) không bị mất mẫu
    stock = yf.download(ticker, period="130d", progress=False)
    if stock.empty:
        return None, None

    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = stock.columns.get_level_values(0)

    stock['Volume'] = np.log1p(stock['Volume'])

    # --- Chỉ báo kỹ thuật gốc ---
    stock['SMA_10'] = stock['Close'].rolling(10).mean()
    stock['SMA_20'] = stock['Close'].rolling(20).mean()
    stock['EMA_20'] = stock['Close'].ewm(span=20, adjust=False).mean()

    delta = stock['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    stock['RSI_14'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    ema12 = stock['Close'].ewm(span=12, adjust=False).mean()
    ema26 = stock['Close'].ewm(span=26, adjust=False).mean()
    stock['MACD']        = ema12 - ema26
    stock['Signal_Line'] = stock['MACD'].ewm(span=9, adjust=False).mean()

    std20 = stock['Close'].rolling(20).std()
    stock['BB_Middle'] = stock['Close'].rolling(20).mean()
    stock['BB_Upper']  = stock['BB_Middle'] + std20 * 2
    stock['BB_Lower']  = stock['BB_Middle'] - std20 * 2

    # ── [MỚI] ĐỒNG BỘ DỮ LIỆU VĨ MÔ THỜI GIAN THỰC ĐỂ INFERENCE ───────────
    try:
        # Tải chỉ số vĩ mô thời gian thực cho chuỗi ngày tương ứng
        macro_data = yf.download(['^VIX', '^TNX'], period="130d", progress=False)
        if isinstance(macro_data.columns, pd.MultiIndex):
            macro_data.columns = [f"{col[0]}_{col[1]}" for col in macro_data.columns]
            
        stock['VIX'] = macro_data['Close_^VIX']
        stock['TNX'] = macro_data['Close_^TNX']
    except Exception as e:
        logger.warning(f"Không tải được dữ liệu vĩ mô thời gian thực, đang dùng fallback. Chi tiết: {e}")
        stock['VIX'] = 20.0   # Điểm trung bình lịch sử của VIX
        stock['TNX'] = 4.0    # Điểm trung bình lãi suất

    # Giả lập điểm tin tức hoặc đọc từ API tin tức bên ngoài (Ví dụ: tích hợp Finnhub)
    # Ví dụ mẫu: Lấy ngẫu nhiên từ [-0.2, 0.4] hoặc bạn có thể gán cứng 0.0 (Trung lập)
    stock['Sentiment_Score'] = 0.0 
    
    # Điền khuyết dữ liệu nếu lệch múi giờ giao dịch
    stock = stock.ffill().bfill()
    stock.dropna(inplace=True)
    
    return stock, float(stock['Close'].iloc[-1])


# ==========================================
# 5. ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/cache/status', methods=['GET'])
def cache_status():
    with _cache_lock:
        entries = []
        now = time.time()
        for ticker, data in _stock_cache.items():
            age_s    = int(now - data["fetched_at"])
            ttl_left = max(0, 3600 - age_s)
            entries.append({
                "ticker"    : ticker,
                "age_s"     : age_s,
                "ttl_left_s": ttl_left,
                "ttl_left"  : f"{ttl_left // 60}m {ttl_left % 60}s",
            })
    logger.info(f"Yêu cầu kiểm tra trạng thái cache. Kích thước hiện tại: {len(entries)}/50")
    return jsonify({
        "cache_size"    : len(entries),
        "cache_maxsize" : 50,
        "cache_ttl_s"   : 3600,
        "entries"       : entries
    })


@app.route('/cache/clear', methods=['POST'])
def cache_clear():
    with _cache_lock:
        count = len(_stock_cache)
        _stock_cache.clear()
    logger.warning(f"Lệnh Force Refresh: Đã dọn sạch toàn bộ cache hệ thống ({count} mã bị xóa).")
    return jsonify({"cleared": count, "status": "ok"})


@app.route('/predict', methods=['POST'])
@limiter.limit(", ".join(PREDICT_LIMITS))
def predict():
    req_id = np.random.randint(100000, 999999)  # Tạo Request ID ngẫu nhiên để truy vết log
    start_time = time.time()

    if model is None:
        logger.error(f"[REQ-{req_id}] Từ chối xử lý: Mô hình AI chưa được load thành công!")
        return jsonify({"error": "Mô hình chưa được load thành công!"}), 500

    try:
        data   = request.get_json() or {}
        ticker = data.get('ticker', 'AAPL').upper()

        n_samples  = int(data.get('n_samples', 100))
        confidence = float(data.get('confidence', 0.90))

        # Giới hạn an toàn tham số
        n_samples  = max(50, min(n_samples, 500))
        confidence = max(0.50, min(confidence, 0.99))

        logger.info(f"[REQ-{req_id}] Nhận lệnh dự báo | Mã: {ticker} | MC Samples: {n_samples} | Confidence: {confidence*100}%")

        # Đọc dữ liệu (tự động điều hướng cache/yfinance trong hàm)
        stock, last_close, from_cache = get_cached_stock(ticker)

        if stock is None:
            logger.warning(f"[REQ-{req_id}] Thất bại: Không tìm thấy hoặc không tải được dữ liệu cho mã '{ticker}'")
            return jsonify({"error": f"Không tìm thấy mã {ticker}!"}), 404

        recent_data = stock.tail(WINDOW_SIZE)
        if len(recent_data) < WINDOW_SIZE:
            logger.warning(f"[REQ-{req_id}] Thất bại: Tập dữ liệu của {ticker} chỉ có {len(recent_data)} hàng (Yêu cầu: {WINDOW_SIZE})")
            return jsonify({"error": "Dữ liệu không đủ!"}), 400

        # --- Chuẩn bị tensor ---
        scaled   = scaler_x.transform(recent_data[FEATURE_COLS].values)
        tensor_x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(device)

        # --- MC Dropout inference ---
        mc = predict_with_uncertainty(tensor_x, n_samples=n_samples, confidence=confidence)

        # --- Chuyển % thay đổi → giá USD ---
        predicted_price = round(last_close * (1 + mc["mean_return"]),  2)
        lower_price     = round(last_close * (1 + mc["lower_return"]), 2)
        upper_price     = round(last_close * (1 + mc["upper_return"]), 2)

        next_date = (recent_data.index[-1] + pd.offsets.BDay(1)).strftime('%d/%m/%Y')
        latency = time.time() - start_time

        # LOGGING THÀNH CÔNG: Ghi nhận toàn bộ thông số phân phối và độ trễ vào file log
        logger.info(
            f"[REQ-{req_id}] THÀNH CÔNG | Mã: {ticker} | Cache: {from_cache} | "
            f"Giá cuối: {last_close:.2f} -> Dự báo: {predicted_price:.2f} ({mc['mean_return']*100:+.2f}%) | "
            f"Khoảng giá [{confidence*100}%]: ({lower_price:.2f} sang {upper_price:.2f}) | "
            f"Độ bất định (Std): {mc['std_return']*100:.4f}% | Thời gian tính: {latency:.4f}s"
        )

        return jsonify({
            "ticker"              : ticker,
            "predict_date"        : next_date,
            "last_close"          : round(last_close, 2),
            "predicted_price"     : predicted_price,
            "predicted_change_pct": round(mc["mean_return"] * 100, 2),
            "confidence"          : confidence,
            "lower_price"         : lower_price,
            "upper_price"         : upper_price,
            "lower_change_pct"    : round(mc["lower_return"] * 100, 2),
            "upper_change_pct"    : round(mc["upper_return"] * 100, 2),
            "uncertainty_std"     : round(mc["std_return"] * 100, 4),
            "n_samples"           : n_samples,
            "distribution"        : mc["all_samples"],
            "cache_hit"           : from_cache,
            "status"              : "success"
        })

    except Exception as e:
        latency = time.time() - start_time
        # LOGGING LỖI: Ghi lại chi tiết lỗi hệ thống (như lỗi Tensor, OOM bộ nhớ GPU, hoặc lỗi chuyển đổi) kèm Traceback cụ thể
        logger.error(f"[REQ-{req_id}] LỖI HỆ THỐNG | Thời gian chạy trước lỗi: {latency:.4f}s | Chi tiết lỗi: {str(e)}", exc_info=True)
        return jsonify({"error": f"Lỗi xử lý hệ thống nội bộ."}), 400


# ==========================================
# XỬ LÝ LỖI RATE LIMIT — trả về JSON thay vì HTML mặc định
# ==========================================
@app.errorhandler(429)
def ratelimit_handler(e):
    logger.warning(f"Phát hiện cảnh báo chặn IP [{get_remote_address()}]: Vượt ngưỡng giới hạn yêu cầu (Rate Limit).")
    return jsonify({
        "error"      : "Quá nhiều yêu cầu! Vui lòng thử lại sau.",
        "retry_after": e.description,
        "status"     : 429
    }), 429


if __name__ == '__main__':
    # Trong môi trường production thật, cân nhắc đổi sang gunicorn/uwsgi thay vì app.run(debug=True)
    app.run(debug=True, port=5000)