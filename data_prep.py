import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_data(ticker, start_date, end_date, window_size=16):
    # 1. Tải dữ liệu
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Xử lý nếu yfinance trả về MultiIndex (thường gặp ở các bản mới)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. Tính toán Feature Engineering
    # Moving Average 20 ngày
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. Làm sạch dữ liệu (Xóa các dòng NaN do tính toán rolling)
    df.dropna(inplace=True)
    
    # 4. Chọn các cột đặc trưng (Features) và Mục tiêu (Target)
    # CẬP NHẬT: Thêm Open, High, Low để tạo bộ 7 biến (OHLCV + MA20 + RSI)
    features_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'RSI']
    features = df[features_cols].values
    target = df['Close'].values.reshape(-1, 1)

    # 5. Chuẩn hóa dữ liệu (Mỗi cột chuẩn hóa riêng biệt)
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    scaled_features = scaler_x.fit_transform(features)
    scaled_target = scaler_y.fit_transform(target)

    # 6. Tạo Sliding Window (Cấu trúc dữ liệu cho mạng nơ-ron)
    X, y = [], []
    for i in range(window_size, len(scaled_features)):
        # Lấy 'window_size' bước thời gian của TẤT CẢ 7 features
        X.append(scaled_features[i-window_size:i]) 
        # Dự báo giá Close của ngày tiếp theo
        y.append(scaled_target[i]) 
        
    X, y = np.array(X), np.array(y)
    
    # 7. Chia tập Train/Test (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # In ra để kiểm tra
    print(f"Kích thước X_train: {X_train.shape} (Mẫu, Thời gian, Đặc trưng)")
    print(f"Kích thước y_train: {y_train.shape}")
    
    return X_train, y_train, X_test, y_test, scaler_y