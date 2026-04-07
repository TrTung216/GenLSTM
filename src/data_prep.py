import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler, StandardScaler # [SỬA 1]: Thêm StandardScaler

def add_technical_indicators(df):
    df = df.copy()
    # 1. Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # 2. RSI_14
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    ema_gain = gain.ewm(com=13, adjust=False).mean()
    ema_loss = loss.ewm(com=13, adjust=False).mean()
    df['RSI_14'] = 100 - (100 / (1 + (ema_gain / (ema_loss + 1e-9))))

    # 3. MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 4. Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    std_dev = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (std_dev * 2)
    df['BB_Lower'] = df['BB_Middle'] - (std_dev * 2)

    df.dropna(inplace=True)
    return df

def download_stock_data(ticker, start_date, end_date):
    print(f"--- Đang tải dữ liệu cho {ticker} ---")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if df.empty or len(df) < 100:
        return None
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    return df

def prepare_data_from_df(df_input, window_size=16):
    if df_input is None: return None, None, None, None, None
    
    df = df_input.copy()
    
    # 1. Log Volume
    df['Volume'] = np.log1p(df['Volume'])

    # 2. Indicators
    df = add_technical_indicators(df)
    
    df['Target_Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)

    # 3. Features & Target
    features_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'SMA_10', 'SMA_20', 'EMA_20', 'RSI_14', 
        'MACD', 'Signal_Line', 'BB_Middle', 'BB_Upper', 'BB_Lower'
    ]
    
    # Kiểm tra số lượng mẫu sau khi dropna
    if len(df) <= window_size:
        return None, None, None, None, None
    features = df[features_cols].values
    target = df['Target_Return'].values.reshape(-1, 1)

    # 4. Scaling
    scaler_x = RobustScaler()
    scaler_y = StandardScaler()
    scaled_features = scaler_x.fit_transform(features)
    scaled_target = scaler_y.fit_transform(target)

    # Lưu scalers
    joblib.dump(scaler_x, 'scaler_x.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')
    
    # 5. Sliding Window
    X, y = [], []
    for i in range(window_size, len(scaled_features)):
        X.append(scaled_features[i-window_size:i]) 
        y.append(scaled_target[i]) 
        
    X, y = np.array(X), np.array(y)
    
    # 6. Train/Test Split
    split = int(len(X) * 0.8)
    return X[:split], y[:split], X[split:], y[split:], scaler_y