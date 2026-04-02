import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ
# ==========================================
TICKER_SYMBOL = "AAPL"
START_DATE = "2015-01-01"
END_DATE = "2026-01-01"
WINDOW_SIZE = 60  # Dùng 60 ngày quá khứ để dự đoán 1 ngày tương lai

print(f"Đang tải dữ liệu cho mã: {TICKER_SYMBOL}...")

# ==========================================
# 2. TẢI VÀ LÀM SẠCH DỮ LIỆU
# ==========================================
# Tải dữ liệu từ Yahoo Finance
df = yf.download(TICKER_SYMBOL, start=START_DATE, end=END_DATE)

# 🛑 FIX LỖI: Xử lý cấu trúc MultiIndex của yfinance phiên bản mới
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1) # Gỡ bỏ tầng tên cổ phiếu (AAPL), chỉ giữ lại 'Close'

# Lấy an toàn cột 'Close' và đưa về mảng 2 chiều
data = df[['Close']].values

# Kiểm tra xem có giá trị NaN nào không và loại bỏ
if np.isnan(data).any():
    print("Phát hiện dữ liệu bị thiếu (NaN), đang tiến hành xử lý...")
    df = df.dropna()
    data = df[['Close']].values

print(f"Tổng số ngày giao dịch tải về: {len(data)}")

# ==========================================
# 3. CHUẨN HÓA DỮ LIỆU (NORMALIZATION)
# ==========================================
# Scale dữ liệu về khoảng (0, 1) giúp LSTM hội tụ nhanh hơn
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# ==========================================
# 4. TẠO CẤU TRÚC CỬA SỔ TRƯỢT (SLIDING WINDOW)
# ==========================================
X = []
y = []

# Trượt qua dữ liệu để tạo các cặp (Input = 60 ngày, Output = ngày thứ 61)
for i in range(WINDOW_SIZE, len(scaled_data)):
    X.append(scaled_data[i-WINDOW_SIZE:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# Định hình lại X thành 3D (Samples, Time Steps, Features) cho LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# ==========================================
# 5. CHIA TẬP TRAIN / TEST
# ==========================================
# Chia 80% để Train, 20% để Test
train_size = int(len(X) * 0.8)

X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

print(f"Kích thước tập X_train: {X_train.shape}")
print(f"Kích thước tập y_train: {y_train.shape}")
print(f"Kích thước tập X_test: {X_test.shape}")

# Vẽ biểu đồ giá thực tế để kiểm tra
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label=f'Giá đóng cửa {TICKER_SYMBOL}', color='blue')
plt.title(f'Lịch sử giá cổ phiếu {TICKER_SYMBOL}')
plt.xlabel('Thời gian')
plt.ylabel('Giá (USD)')
plt.legend()
plt.grid(True)
plt.show()