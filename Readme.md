# GenLSTM

GenLSTM là dự án dự báo giá cổ phiếu cho phiên giao dịch kế tiếp bằng mô hình `CNN-LSTM` có `attention`, được tối ưu siêu tham số bằng `GA-WOA` và ước lượng độ bất định bằng `Monte Carlo Dropout`.

Repo hiện có hai phần chính:

- `src/ga_lstm.py`: huấn luyện và tìm siêu tham số tốt nhất.
- `app.py`: web app Flask để suy luận nhanh theo mã cổ phiếu.

## Tính năng chính

- Tối ưu siêu tham số tự động bằng lai ghép giữa `Genetic Algorithm` và `Whale Optimization Algorithm`.
- Mô hình chuỗi thời gian dùng `1D-CNN + LSTM + Attention`.
- Dữ liệu đầu vào gồm `OHLCV` và các chỉ báo kỹ thuật như `SMA`, `EMA`, `RSI`, `MACD`, `Bollinger Bands`.
- Mục tiêu dự báo là `tỷ suất sinh lời ngày kế tiếp`, sau đó quy đổi ngược ra giá dự báo.
- Web app hỗ trợ:
  - chọn ticker
  - điều chỉnh số lần Monte Carlo sampling
  - chọn mức confidence interval
  - xem phân phối dự báo, độ lệch chuẩn và khoảng tin cậy
- Có `TTL cache` cho dữ liệu `yfinance` và `rate limit` cho API suy luận.

## Cấu trúc thư mục

```text
GenLSTM/
├── app.py
├── Dockerfile
├── artifacts/
│   ├── best_model.pth
│   ├── model_config.json
│   ├── scaler_x.pkl
│   └── scaler_y.pkl
├── requirements.txt
├── templates/
│   └── index.html
├── src/
│   ├── data_prep.py
│   ├── fitness_function.py
│   └── ga_lstm.py
├── References/
└── Reports/
```

## Pipeline hoạt động

1. Tải dữ liệu giá từ `Yahoo Finance`.
2. Làm sạch dữ liệu, log-transform `Volume`, tính chỉ báo kỹ thuật.
3. Tạo `sliding window` cho bài toán dự báo chuỗi thời gian.
4. Dùng `GA-WOA` để tìm bộ siêu tham số tốt nhất cho mô hình.
5. Huấn luyện mô hình cuối cùng với bộ tham số tối ưu.
6. Lưu artifact để web app suy luận:
   - `best_model.pth`
   - `scaler_x.pkl`
   - `scaler_y.pkl`
   - `model_config.json`
7. Khi gọi `/predict`, app tải dữ liệu gần nhất, dựng feature, chạy `MC Dropout` nhiều lần và trả về:
   - giá dự báo trung bình
   - phần trăm thay đổi dự báo
   - khoảng tin cậy
   - độ bất định
   - phân phối mẫu dự báo

## Yêu cầu môi trường

- Python `3.10+` được khuyến nghị.
- Có kết nối Internet để gọi `yfinance`.
- GPU CUDA là tùy chọn, không bắt buộc.

## Cài đặt

Ví dụ trên PowerShell:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install matplotlib pandas_market_calendars
```

Ghi chú:

- `requirements.txt` hiện đủ cho phần web app.
- Phần huấn luyện trong `src/ga_lstm.py` còn cần thêm `matplotlib` và `pandas_market_calendars`, nên phải cài bổ sung như trên.
- Nếu muốn chạy bằng GPU, nên cài bản `PyTorch` phù hợp với CUDA của máy theo hướng dẫn chính thức trước khi cài phần còn lại của dependency.

## Huấn luyện mô hình

Chạy từ thư mục gốc của repo:

```bash
python -m src.ga_lstm
```

Script sẽ:

- tải dữ liệu `AAPL` từ `2015-01-01` tới ngày giao dịch NYSE gần nhất
- chạy vòng lặp `GA-WOA`
- huấn luyện mô hình cuối cùng
- lưu artifact suy luận vào thư mục gốc
- xuất biểu đồ hội tụ `ga_convergence.png`

Lưu ý:

- `app.py` phụ thuộc trực tiếp vào các artifact đã huấn luyện.
- Nếu chưa có `best_model.pth`, `scaler_x.pkl` hoặc `scaler_y.pkl`, API `/predict` sẽ không hoạt động đúng.

## Chạy web app

Sau khi đã có artifact huấn luyện, khởi động Flask app từ thư mục gốc:

```bash
python app.py
```

Mở trình duyệt tại:

```text
http://127.0.0.1:5000
```

Giao diện cho phép nhập mã cổ phiếu, số mẫu Monte Carlo và mức confidence interval để xem dự báo cho phiên kế tiếp.

## API chính

### `POST /predict`

Ví dụ request trên PowerShell:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:5000/predict `
  -ContentType "application/json" `
  -Body '{"ticker":"AAPL","n_samples":100,"confidence":0.90}'
```

Input:

- `ticker`: mã cổ phiếu, mặc định `AAPL`
- `n_samples`: số lần lấy mẫu MC Dropout, bị chặn trong khoảng `50` tới `500`
- `confidence`: mức tin cậy, bị chặn trong khoảng `0.50` tới `0.99`

Output tiêu biểu:

- `predicted_price`
- `predicted_change_pct`
- `lower_price`
- `upper_price`
- `uncertainty_std`
- `distribution`
- `cache_hit`

### `GET /cache/status`

Trả về trạng thái cache dữ liệu ticker, gồm số entry hiện có và TTL còn lại.

### `POST /cache/clear`

Xóa toàn bộ cache dữ liệu đã lưu trong RAM.

## Rate limit và cache

- Giới hạn mặc định cho toàn app:
  - `200 request/ngày`
  - `50 request/giờ`
- Giới hạn riêng cho `/predict`:
  - `10 request/phút`
  - `100 request/ngày`
- Cache dữ liệu `yfinance`:
  - tối đa `50` ticker
  - TTL `3600` giây

## Kết quả thực nghiệm Phase 1

Phase 1 sử dụng 5 random seeds (`42, 123, 2026, 7, 99`) và cùng **587 held-out target observations**. Train/validation/test được cố định trên raw timeline trước khi tạo sliding window. GA-WOA sử dụng 3 expanding walk-forward folds với scaler được fit riêng trên training history của từng fold; test set không tham gia hyperparameter search hoặc early stopping.

Sau khi hoàn thiện protocol chống leakage, GA-WOA được chạy lại từ đầu và chọn cấu hình:

```text
Units=32, Dropout=0.05, LR=0.001, Batch=16,
Window=20, CNN Filters=64, LSTM Layers=1
```

| Model | RMSE (mean ± std) ↓ | MAE (mean ± std) ↓ | Directional Accuracy (mean ± std) ↑ |
| --- | ---: | ---: | ---: |
| LSTM | 0.018416 ± 0.000196 | 0.012646 ± 0.000250 | 47.78% ± 1.39% |
| **CNN-LSTM** | **0.018083 ± 0.000081** | 0.012222 ± 0.000092 | 47.34% ± 0.95% |
| CNN-LSTM-Attention | 0.018139 ± 0.000136 | 0.012321 ± 0.000183 | 46.76% ± 1.22% |
| GA-WOA CNN-LSTM-Attention | 0.018088 ± 0.000142 | **0.012220 ± 0.000183** | **47.92% ± 1.22%** |

Kết quả cuối không cho thấy một mô hình thắng toàn bộ metric. CNN-LSTM có mean RMSE thấp nhất; cấu hình GA-WOA có mean MAE thấp nhất và Directional Accuracy cao nhất, nhưng chênh lệch error giữa hai mô hình rất nhỏ. Directional Accuracy trung bình của tất cả cấu hình vẫn dưới 50%, vì vậy Phase 1 không đưa ra kết luận về khả năng dự báo hướng giá đáng tin cậy.

### Controlled architecture ablation

Ba kiến trúc dưới đây được huấn luyện bằng **cùng cấu hình GA-WOA cuối**, cùng 5 seeds và cùng test targets:

| Architecture | RMSE ↓ | MAE ↓ | Directional Accuracy ↑ |
| --- | ---: | ---: | ---: |
| LSTM | 0.018143 ± 0.000048 | 0.012312 ± 0.000112 | 47.65% ± 2.46% |
| **CNN-LSTM** | **0.017944 ± 0.000070** | **0.012084 ± 0.000092** | **48.09% ± 2.07%** |
| CNN-LSTM-Attention | 0.018026 ± 0.000078 | 0.012144 ± 0.000108 | **48.09% ± 1.13%** |

Controlled ablation cho thấy CNN cải thiện RMSE/MAE so với LSTM trong cấu hình này, còn Attention không tạo thêm cải thiện ổn định so với CNN-LSTM.

<p align="center">
  <img src="experiments/plots/model_rmse_comparison.png" width="48%" alt="RMSE comparison">
  <img src="experiments/plots/model_mae_comparison.png" width="48%" alt="MAE comparison">
</p>

<p align="center">
  <img src="experiments/plots/model_directional_accuracy.png" width="58%" alt="Directional accuracy comparison">
</p>

### Fitness ablation

Fitness-profile ablation A/B/C được giữ lại như **exploratory analysis** vì mỗi profile chỉ được search một lần và experiment này có trước final full-protocol rerun. Không dùng kết quả đó để khẳng định một objective tốt nhất tổng quát.

### Tài liệu nghiên cứu

Chi tiết methodology, kết quả, thảo luận và threats to validity nằm trong:

- `Reports/METHODOLOGY.md`
- `Reports/RESULTS.md`
- `Reports/DISCUSSION.md`
- `Reports/LIMITATIONS.md`

### Reproduce

```bash
python -m pytest -q
python -m src.ga_lstm
python -m src.benchmark
python -m src.architecture_ablation
```

### GA-WOA convergence

<p align="center">
  <img src="experiments/plots/ga_convergence.png" width="70%" alt="GA-WOA convergence">
</p>

## Hạn chế hiện tại

- Dự án hiện dự báo `1 bước tiếp theo`, chưa hỗ trợ dự báo nhiều phiên liên tiếp.
- Chất lượng dự báo phụ thuộc mạnh vào dữ liệu `yfinance` và giai đoạn thị trường.
- Web app chỉ suy luận được khi artifact huấn luyện và cấu hình đang đồng bộ với nhau.
- `Dockerfile` hiện chạy bằng `gunicorn`, vì vậy môi trường container cần có `gunicorn` trước khi dùng flow Docker.

## Tài liệu tham khảo

Các tài liệu nghiên cứu được lưu trong thư mục [`References/`](References/), dùng để tham chiếu cho hướng tiếp cận `GA-LSTM`, `GA-WOA-LSTM` và tối ưu dự báo chuỗi thời gian tài chính.

## Cảnh báo

Kết quả của dự án chỉ phục vụ mục đích học thuật và thử nghiệm kỹ thuật. Không nên dùng trực tiếp để ra quyết định đầu tư thực tế.
