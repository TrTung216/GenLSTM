# Sử dụng Python bản nhẹ
FROM python:3.10-slim

# --- CÁC BIẾN MÔI TRƯỜNG TỐI ƯU CHO PYTHON ---
ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

# --- BẢO MẬT: TẠO NON-ROOT USER ---
# Tạo một user mới tên là "appuser" không có mật khẩu
RUN adduser --disabled-password --gecos '' appuser

# Thiết lập thư mục làm việc
WORKDIR /app

# --- CÀI ĐẶT THƯ VIỆN ---
COPY requirements.txt .
# Nâng cấp pip và cài đặt thư viện
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# --- BẢO MẬT: CHUYỂN QUYỀN SỞ HỮU VÀ ĐỔI USER ---
# Đổi chủ sở hữu của thư mục /app sang cho appuser
RUN chown -R appuser:appuser /app
# Chuyển sang sử dụng user vừa tạo thay vì root
USER appuser

# Mở cổng 5000
EXPOSE 5000

# Khởi chạy server bằng Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]