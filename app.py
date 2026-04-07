from flask import Flask, render_template, request, jsonify
import yfinance as yf
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import json
import traceback

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. ĐỊNH NGHĨA CẤU TRÚC AI (GIỮ NGUYÊN)
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
# 2. HÀM LOAD TỰ ĐỘNG
# ==========================================
def load_system():
    try:
        with open('model_config.json', 'r') as f:
            config = json.load(f)
        
        net = CNN_LSTM(
            input_size=config['input_size'],
            hidden_layer_size=config['hidden_layer_size'],
            dropout_rate=config['dropout_rate'],
            cnn_filters=config['cnn_filters'],
            num_layers=config['num_layers']
        )
        
        net.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
        net.to(device)
        net.eval()

        sx = joblib.load('scaler_x.pkl')
        sy = joblib.load('scaler_y.pkl')
        
        print(f"✅ Hệ thống sẵn sàng! Window: {config['window_size']}")
        return net, sx, sy, config['window_size']
    
    except Exception as e:
        print(f"❌ Lỗi khởi động hệ thống: {e}")
        return None, None, None, 16

model, scaler_x, scaler_y, WINDOW_SIZE = load_system()

# ==========================================
# 3. XỬ LÝ LOGIC DỰ BÁO (CẬP NHẬT RETURN)
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Mô hình chưa được load thành công!"}), 500

    data = request.get_json()
    ticker = data.get('ticker', 'AAPL').upper()
    
    try:
        stock = yf.download(ticker, period="100d", progress=False)
        if stock.empty:
            return jsonify({"error": f"Không tìm thấy mã {ticker}!"}), 404

        if isinstance(stock.columns, pd.MultiIndex):
            stock.columns = stock.columns.get_level_values(0)

        # Tiền xử lý giống hệt lúc Train
        stock['Volume'] = np.log1p(stock['Volume'])

        # Chỉ báo kỹ thuật
        stock['SMA_10'] = stock['Close'].rolling(window=10).mean()
        stock['SMA_20'] = stock['Close'].rolling(window=20).mean()
        stock['EMA_20'] = stock['Close'].ewm(span=20, adjust=False).mean()

        delta = stock['Close'].diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14).mean()
        stock['RSI_14'] = 100 - (100 / (1 + gain/(loss + 1e-9)))

        ema12 = stock['Close'].ewm(span=12, adjust=False).mean()
        ema26 = stock['Close'].ewm(span=26, adjust=False).mean()
        stock['MACD'] = ema12 - ema26
        stock['Signal_Line'] = stock['MACD'].ewm(span=9, adjust=False).mean()

        std20 = stock['Close'].rolling(window=20).std()
        stock['BB_Middle'] = stock['Close'].rolling(window=20).mean()
        stock['BB_Upper'] = stock['BB_Middle'] + (std20 * 2)
        stock['BB_Lower'] = stock['BB_Middle'] - (std20 * 2)

        stock.dropna(inplace=True)

        features_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume', 
            'SMA_10', 'SMA_20', 'EMA_20', 'RSI_14', 
            'MACD', 'Signal_Line', 'BB_Middle', 'BB_Upper', 'BB_Lower'
        ]
        
        recent_data = stock.tail(WINDOW_SIZE)
        if len(recent_data) < WINDOW_SIZE:
            return jsonify({"error": "Dữ liệu không đủ!"}), 400

        # --- DỰ BÁO ---
        scaled_features = scaler_x.transform(recent_data[features_cols].values)
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_scaled = model(input_tensor).cpu().numpy()

        # 1. Giải mã ra % thay đổi dự báo
        predicted_return = float(scaler_y.inverse_transform(pred_scaled)[0][0])
        
        # 2. Lấy giá đóng cửa cuối cùng thực tế
        last_close_price = float(recent_data['Close'].iloc[-1])
        
        # 3. Tính toán giá USD dự báo
        final_price = round(last_close_price * (1 + predicted_return), 2)
        
        # Tính toán % hiển thị (để Web hiển thị thêm cho trực quan)
        change_pct = round(predicted_return * 100, 2)

        next_date = (recent_data.index[-1] + pd.offsets.BDay(1)).strftime('%d/%m/%Y')

        return jsonify({
            "ticker": ticker,
            "predicted_price": final_price,
            "predicted_change_pct": change_pct, # Trả về thêm % tăng giảm
            "last_close": round(last_close_price, 2),
            "predict_date": next_date,
            "status": "success"
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Lỗi xử lý: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)