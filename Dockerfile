# Sử dụng Python bản nhẹ
FROM python:3.10-slim

# Thiết lập thư mục làm việc trong Container
WORKDIR /app

# Copy requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào Container
COPY . .

# Mở cổng 5000 cho Flask
EXPOSE 5000

# Khởi chạy server bằng Gunicorn (cho production)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]