import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_data(ticker, start_date, end_date, window_size=60):
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
    
    # Lấy thêm Volume (đã có sẵn trong df)
    
    # 3. Làm sạch dữ liệu (Xóa các dòng NaN do tính toán rolling)
    df.dropna(inplace=True)
    
    # 4. Chọn các cột đặc trưng (Features) và Mục tiêu (Target)
    # Chúng ta dùng: Close, MA20, RSI, Volume làm đầu vào
    features = df[['Close', 'MA20', 'RSI', 'Volume']].values
    target = df['Close'].values.reshape(-1, 1)

    # 5. Chuẩn hóa dữ liệu (Mỗi cột chuẩn hóa riêng biệt)
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    scaled_features = scaler_x.fit_transform(features)
    scaled_target = scaler_y.fit_transform(target)

    # 6. Tạo Sliding Window (Cấu trúc dữ liệu cho LSTM)
    X, y = [], []
    for i in range(window_size, len(scaled_features)):
        X.append(scaled_features[i-window_size:i]) # Lấy 60 bước thời gian của tất cả features
        y.append(scaled_target[i]) # Dự báo giá Close của ngày tiếp theo
        
    X, y = np.array(X), np.array(y)
    
    # 7. Chia tập Train/Test (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, y_train, X_test, y_test, scaler_y