from flask import Flask, render_template, request, jsonify
# from src.ga_lstm import predict_stock_price # Cần đóng gói hàm dự đoán để gọi ở đây

app = Flask(__name__)

@app.route('/')
def home():
    # Render giao diện web
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Lấy mã cổ phiếu từ request của user
    data = request.get_json()
    ticker = data.get('ticker', 'AAPL')
    
    # Tại đây sẽ gọi hàm dự đoán từ thư mục src/
    # result = predict_stock_price(ticker)
    
    # Tạm thời trả về dữ liệu mẫu
    return jsonify({
        "status": "success",
        "ticker": ticker,
        "predicted_price": 150.50,
        "message": "API đang hoạt động!"
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)