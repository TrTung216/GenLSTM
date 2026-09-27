import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
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
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(scaler_x, "artifacts/scaler_x.pkl")
        joblib.dump(scaler_y, "artifacts/scaler_y.pkl")

    # ── Sliding Window → 3D (n, window, features) ────────────────────────────
    X, y = [], []
    for i in range(window_size, len(scaled_features)):
        X.append(scaled_features[i - window_size:i])
        y.append(scaled_target[i])

    X = np.array(X)
    y = np.array(y).flatten()

    # ── Train / Test Split 80/20 ─────────────────────────────────────────────
    split   = int(len(X) * 0.8)
    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]

    return X_train, y_train, X_test, y_test, scaler_y


def prepare_train_validation_test(
    df_input,
    window_size=16,
    validation_fraction=0.15,
    test_fraction=0.20,
    save_scalers=False,
    scaler_fit_fraction=None,
):
    """Build explicit chronological partitions with train-only scalers.

    The scaler cut is based on the target position of each sequence, rather
    than an approximate raw-data percentage.  Validation and test statistics
    therefore never influence fitted preprocessing parameters.
    """
    from src.validation import temporal_split_indices

    if df_input is None or df_input.empty:
        return (None,) * 8
    df = df_input.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['Volume'] = np.log1p(df['Volume'])
    df = add_technical_indicators(df)
    df['Sentiment_Score'] = compute_cmf(df, period=20)
    df['Target_Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    n_sequences = len(df) - window_size
    if n_sequences < 3:
        return (None,) * 8

    split = temporal_split_indices(n_sequences, validation_fraction, test_fraction)
    # A sequence numbered k predicts raw row window_size + k. By default the
    # scaler sees the outer training partition. During walk-forward search the
    # caller supplies the initial-fold fraction so even the first validation
    # fold remains unseen by preprocessing.
    if scaler_fit_fraction is None:
        scaler_sequence_end = split.train.stop
    else:
        if not 0 < scaler_fit_fraction <= 1:
            raise ValueError("scaler_fit_fraction must be in (0, 1]")
        development_end = split.validation.stop
        scaler_sequence_end = max(1, int(development_end * scaler_fit_fraction))
    train_raw_end = window_size + scaler_sequence_end
    features = df[FEATURE_COLS].to_numpy()
    target = df['Target_Return'].to_numpy().reshape(-1, 1)
    scaler_x, scaler_y = RobustScaler(), StandardScaler()
    scaler_x.fit(features[:train_raw_end])
    scaler_y.fit(target[:train_raw_end])
    features = scaler_x.transform(features)
    target = scaler_y.transform(target).flatten()

    X = np.asarray([features[i - window_size:i] for i in range(window_size, len(df))])
    y = target[window_size:]
    if save_scalers:
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(scaler_x, "artifacts/scaler_x.pkl")
        joblib.dump(scaler_y, "artifacts/scaler_y.pkl")
    return (
        X[split.train], y[split.train],
        X[split.validation], y[split.validation],
        X[split.test], y[split.test],
        scaler_x, scaler_y,
    )


def prepare_walk_forward_folds(
    df_input,
    window_size=16,
    n_splits=3,
    validation_fraction=0.15,
    test_fraction=0.20,
):
    """Build walk-forward folds with a scaler fitted independently per fold.

    Only the outer development partition (train + validation) is considered.
    For every expanding fold, RobustScaler/StandardScaler are fitted on raw
    rows available up to that fold's training boundary, then used to transform
    that fold's train and validation sequences.  The final test partition is
    never used during GA-WOA search.
    """
    from src.validation import temporal_split_indices, walk_forward_splits

    if df_input is None or df_input.empty:
        return []

    df = df_input.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['Volume'] = np.log1p(df['Volume'])
    df = add_technical_indicators(df)
    df['Sentiment_Score'] = compute_cmf(df, period=20)
    df['Target_Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)

    n_sequences = len(df) - window_size
    if n_sequences < 3:
        return []

    outer = temporal_split_indices(
        n_sequences, validation_fraction, test_fraction,
    )
    development_end = outer.validation.stop
    features_raw = df[FEATURE_COLS].to_numpy()
    target_raw = df['Target_Return'].to_numpy().reshape(-1, 1)

    folds = []
    for train_idx, val_idx in walk_forward_splits(development_end, n_splits):
        # Sequence k predicts raw row window_size + k.  Fit through the final
        # training target row only; validation rows remain completely unseen.
        train_sequence_end = int(train_idx[-1]) + 1
        train_raw_end = window_size + train_sequence_end

        scaler_x = RobustScaler()
        scaler_y = StandardScaler()
        scaler_x.fit(features_raw[:train_raw_end])
        scaler_y.fit(target_raw[:train_raw_end])

        features = scaler_x.transform(features_raw[:window_size + development_end])
        target = scaler_y.transform(
            target_raw[:window_size + development_end]
        ).flatten()

        X = np.asarray([
            features[i - window_size:i]
            for i in range(window_size, window_size + development_end)
        ])
        y = target[window_size:window_size + development_end]

        folds.append((X[train_idx], y[train_idx], X[val_idx], y[val_idx]))

    return folds
