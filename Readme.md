# Tối ưu hóa siêu tham số mô hình LSTM bằng Thuật toán Di truyền (GA-LSTM) trong Dự báo giá Cổ phiếu

Dự án này ứng dụng Thuật toán Di truyền (Genetic Algorithm - GA) để tự động hóa việc tìm kiếm các siêu tham số (Hyperparameters) tối ưu cho mạng nơ-ron LSTM (Long Short-Term Memory). Mô hình cuối cùng được sử dụng để dự báo giá cổ phiếu Apple (AAPL) dựa trên dữ liệu chuỗi thời gian thực tế.

## Điểm nổi bật của dự án
* Tối ưu hóa tự động: Thay vì dò tìm thủ công (Grid Search/Random Search), GA giúp mô hình tự "tiến hóa" để tìm ra cấu trúc LSTM tốt nhất (Units, Dropout, Learning Rate, Batch Size).
* Môi trường PyTorch tối ưu GPU: Hỗ trợ tính toán hiệu năng cao với CUDA trên native Windows, tối đa hóa sức mạnh của Card đồ họa NVIDIA.
* Pipeline dữ liệu tự động: Lấy dữ liệu trực tiếp từ Yahoo Finance, tự động làm sạch và xử lý cấu trúc MultiIndex.
* Kiến trúc Modular: Mã nguồn được chia nhỏ thành các module chuẩn kỹ thuật phần mềm, dễ dàng bảo trì và mở rộng.

## Cấu trúc Repository

```text
GA-LSTM-Stock-Prediction
|-- data_prep.py        # Tải dữ liệu yfinance, tiền xử lý (MinMaxScaler) & tạo Sliding Window.
|-- fitness_function.py # Định nghĩa kiến trúc PyTorch LSTM, huấn luyện (Early Stopping) & tính RMSE.
|-- ga_lstm.py          # Lõi thuật toán GA (Lai ghép, Đột biến) & Main script chạy toàn bộ hệ thống.
|-- README.md           # Tài liệu hướng dẫn.
```

## Cài đặt & Yêu cầu hệ thống

Dự án yêu cầu Python 3.11+. Để sử dụng được GPU NVIDIA (khuyến nghị), bạn cần cài đặt PyTorch phiên bản hỗ trợ CUDA.

1. Clone repository:
```bash
git clone [https://github.com/your-username/GA-LSTM-Stock-Prediction.git](https://github.com/your-username/GA-LSTM-Stock-Prediction.git)
cd GA-LSTM-Stock-Prediction
```

2. Cài đặt thư viện:
```bash
pip install numpy pandas scikit-learn yfinance matplotlib
```

3. Cài đặt PyTorch (Hỗ trợ GPU/CUDA 12.1):
```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

## Hướng dẫn sử dụng

Chỉ cần chạy file thực thi chính ga_lstm.py. Quá trình này sẽ tự động gọi dữ liệu, chạy thuật toán tiến hóa GA, xây dựng mô hình cuối cùng và xuất ra biểu đồ.

```bash
python ga_lstm.py
```

Quá trình thực thi bao gồm các bước:
1. Tải dữ liệu AAPL từ 2015 đến nay.
2. GA khởi tạo quần thể ban đầu và tiến hóa qua các thế hệ. (Theo dõi Terminal để xem chỉ số Fitness tăng dần).
3. Huấn luyện mô hình PyTorch LSTM cuối cùng với bộ gen (siêu tham số) xuất sắc nhất.
4. Vẽ đồ thị sự hội tụ của GA và đồ thị so sánh giá thực tế vs dự đoán.

## Kết quả thực nghiệm


1. Quá trình hội tụ của Thuật toán Di truyền (GA)
Mô hình thể hiện sự cải thiện rõ rệt về điểm số Thích nghi (Fitness) qua từng thế hệ, chứng minh thuật toán đã thoát khỏi các điểm cực tiểu cục bộ.
![GA Convergence](link_to_your_image_1.png)

2. Dự báo Giá cổ phiếu trên tập Test
Mô hình GA-LSTM cuối cùng thể hiện khả năng bám sát các biến động giá (trend) và những điểm đảo chiều quan trọng của thị trường.
![Prediction vs Actual](link_to_your_image_2.png)

## Hướng phát triển tương lai
* Tích hợp thêm Phân tích tâm lý thị trường (Sentiment Analysis) từ dữ liệu tin tức tài chính.
* Thử nghiệm nghiệm thêm các thuật toán tối ưu hóa bầy đàn khác như PSO (Particle Swarm Optimization) hoặc WOA (Whale Optimization Algorithm).
* Đưa thêm các chỉ báo kỹ thuật (RSI, MACD, Bollinger Bands) vào features đầu vào.

## Đóng góp (Contributing)
Mọi đóng góp (Pull Requests) hoặc báo lỗi (Issues) đều được hoan nghênh để cải thiện dự án này.
