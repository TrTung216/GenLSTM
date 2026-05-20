import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler, StandardScaler

# Định nghĩa danh sách các cột đặc trưng mới (Mở rộng từ 14 lên 17 features)
FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume', 
    'SMA_10', 'SMA_20', 'EMA_20', 'RSI_14', 
    'MACD', 'Signal_Line', 'BB_Middle', 'BB_Upper', 'BB_Lower',
    'VIX', 'TNX', 'Sentiment_Score'  # <-- 3 Đặc trưng mới
]

def fetch_macro_data(ticker, start_date, end_date, index_reference):
    """
    Tải VIX + TNX từ yfinance và align theo index của cổ phiếu chính.

    Hàm này chỉ nên gọi MỘT LẦN duy nhất (trong __main__ của ga_lstm.py),
    kết quả được merge vào df_raw trước khi chạy GA — tránh gọi lại
    300+ lần trong vòng lặp chromosome.
    """
    print("  [Macro] Tải VIX + TNX từ yfinance (1 lần duy nhất)...")
    macro_data = yf.download(['^VIX', '^TNX'], start=start_date, end=end_date, progress=False)

    if isinstance(macro_data.columns, pd.MultiIndex):
        macro_data.columns = [f"{col[0]}_{col[1]}" for col in macro_data.columns]

    vix_close      = macro_data.get('Close_^VIX', pd.Series(dtype=float))
    tnx_close      = macro_data.get('Close_^TNX', pd.Series(dtype=float))

    meta_df        = pd.DataFrame(index=index_reference)
    meta_df['VIX'] = vix_close
    meta_df['TNX'] = tnx_close
    meta_df        = meta_df.ffill().bfill()
    return meta_df


def compute_cmf(df, period=20):
    """
    Chaikin Money Flow (CMF) — proxy Sentiment từ OHLCV, không cần API ngoài.

    Ý nghĩa:
      CMF > 0 : dòng tiền vào (tâm lý tích cực, accumulation)
      CMF < 0 : dòng tiền ra (tâm lý tiêu cực, distribution)
      Khoảng [-1, +1], thực tế thường [-0.3, +0.3]

    Tốt hơn Sentiment_Score = 0 vì:
      Giá trị hằng số 0 không mang thông tin nào cho LSTM.
      CMF phản ánh áp lực mua/bán thực tế từ dữ liệu giá có sẵn.
    """
    high_low = df['High'] - df['Low'] + 1e-9
    mfm      = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / high_low
    mfv      = mfm * df['Volume']
    vol_sum  = df['Volume'].rolling(period).sum()
    return mfv.rolling(period).sum() / (vol_sum + 1e-9)

def add_technical_indicators(df):
    df = df.copy()
    # Các chỉ báo kỹ thuật cũ của bạn giữ nguyên 100%
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    ema_gain = gain.ewm(com=13, adjust=False).mean()
    ema_loss = loss.ewm(com=13, adjust=False).mean()
    df['RSI_14'] = 100 - (100 / (1 + (ema_gain / (ema_loss + 1e-9))))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    std20 = df['Close'].rolling(window=20).std()
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Middle'] + std20 * 2
    df['BB_Lower'] = df['BB_Middle'] - std20 * 2
    return df

def prepare_data_from_df(df_input, window_size=16, save_scalers=False):
    """
    Chuẩn bị dữ liệu huấn luyện từ DataFrame thô.

    Yêu cầu: df_input đã có cột VIX, TNX (merge từ fetch_macro_data trước khi gọi).
    Hàm này KHÔNG gọi yfinance nữa — tránh 300+ lần fetch trong GA loop.

    Args:
        df_input     : DataFrame đã có OHLCV + VIX + TNX
        window_size  : lookback steps cho sliding window (gene[4] trong GA)
        save_scalers : True khi train cuối (lưu scaler_x/y.pkl), False trong GA loop

    Returns:
        X_train  : (n_train, window_size, n_features)
        y_train  : (n_train,)
        X_test   : (n_test,  window_size, n_features)
        y_test   : (n_test,)
        scaler_y : StandardScaler đã fit trên train — dùng inverse_transform khi đánh giá
    """
    if df_input is None or df_input.empty:
        return None, None, None, None, None

    df = df_input.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['Volume'] = np.log1p(df['Volume'])
    df = add_technical_indicators(df)

    # ── [FIX 3] Sentiment_Score = CMF proxy thay vì hardcode 0.0 ─────────────
    # CMF tính được từ OHLCV sẵn có, phản ánh áp lực mua/bán thực tế
    # VIX và TNX đã được merge vào df_input từ bên ngoài (ga_lstm.__main__)
    df['Sentiment_Score'] = compute_cmf(df, period=20)
    # ─────────────────────────────────────────────────────────────────────────

    df['Target_Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)

    if len(df) <= window_size:
        return None, None, None, None, None

    features = df[FEATURE_COLS].values
    target   = df['Target_Return'].values.reshape(-1, 1)

    # ── [FIX 2] Chống Data Leakage: fit scaler CHỈ trên tập train ────────────
    # Bản cũ fit trên toàn bộ features (train + test) → scaler "biết trước"
    # thống kê của tập test → đánh giá quá lạc quan.
    raw_split       = int(len(features) * 0.8)
    scaler_x        = RobustScaler()
    scaler_y        = StandardScaler()
    scaler_x.fit(features[:raw_split])          # fit CHỈ trên train
    scaler_y.fit(target[:raw_split])            # fit CHỈ trên train
    scaled_features = scaler_x.transform(features)   # transform toàn bộ
    scaled_target   = scaler_y.transform(target)
    # ─────────────────────────────────────────────────────────────────────────

    if save_scalers:
        joblib.dump(scaler_x, 'scaler_x.pkl')
        joblib.dump(scaler_y, 'scaler_y.pkl')

    # ── Sliding Window → 3D (n, window, features) ────────────────────────────
    X, y = [], []
    for i in range(window_size, len(scaled_features)):
        X.append(scaled_features[i - window_size:i])
        y.append(scaled_target[i])

    X = np.array(X)
    y = np.array(y).flatten()

    # ── Train / Test Split 80/20 ─────────────────────────────────────────────
    split   = int(len(X) * 0.8)
    X_train = X[:split];  y_train = y[:split]
    X_test  = X[split:];  y_test  = y[split:]

    return X_train, y_train, X_test, y_test, scaler_y