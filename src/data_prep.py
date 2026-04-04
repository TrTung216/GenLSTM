import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def add_technical_indicators(df):
    """
    Thêm các chỉ báo kỹ thuật vào DataFrame chứng khoán.
    """
    df = df.copy()

    # 1. Đường trung bình động (Moving Averages)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # 2. RSI (Relative Strength Index - 14 ngày)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    ema_gain = gain.ewm(com=13, adjust=False).mean()
    ema_loss = loss.ewm(com=13, adjust=False).mean()
    rs = ema_gain / ema_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # 3. MACD (Moving Average Convergence Divergence)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 4. Bollinger Bands (20 ngày, 2 Standard Deviations)
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    std_dev = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (std_dev * 2)
    df['BB_Lower'] = df['BB_Middle'] - (std_dev * 2)

    # Xóa các dòng có giá trị NaN do tính toán rolling window (26 ngày đầu tiên)
    df.dropna(inplace=True)

    return df

def prepare_data(ticker, start_date, end_date, window_size=16):
    # 1. Tải dữ liệu
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Xử lý nếu yfinance trả về MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. GỌI HÀM TÍNH TOÁN FEATURE ENGINEERING
    df = add_technical_indicators(df)
    
    # 3. Chọn các cột đặc trưng (Features) và Mục tiêu (Target)
    # CẬP NHẬT: Đưa toàn bộ các cột chỉ báo kỹ thuật mới vào X
    features_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'SMA_10', 'SMA_20', 'EMA_20', 'RSI_14', 
        'MACD', 'Signal_Line', 'BB_Middle', 'BB_Upper', 'BB_Lower'
    ]
    features = df[features_cols].values
    target = df['Close'].values.reshape(-1, 1)

    # 4. Chuẩn hóa dữ liệu (Mỗi cột chuẩn hóa riêng biệt)
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    scaled_features = scaler_x.fit_transform(features)
    scaled_target = scaler_y.fit_transform(target)

    # 5. Tạo Sliding Window (Cấu trúc dữ liệu cho mạng nơ-ron)
    X, y = [], []
    for i in range(window_size, len(scaled_features)):
        # Lấy 'window_size' bước thời gian của TẤT CẢ features
        X.append(scaled_features[i-window_size:i]) 
        # Dự báo giá Close của ngày tiếp theo
        y.append(scaled_target[i]) 
        
    X, y = np.array(X), np.array(y)
    
    # 6. Chia tập Train/Test (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # In ra để kiểm tra
    print(f"Kích thước X_train: {X_train.shape} (Mẫu, Thời gian, Số lượng Đặc trưng)")
    print(f"Kích thước y_train: {y_train.shape}")
    
    return X_train, y_train, X_test, y_test, scaler_y