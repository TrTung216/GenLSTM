BÁO CÁO TIẾN ĐỘ TUẦN
Dự án: Xây dựng Hệ thống Dự báo Giá Cổ phiếu bằng GA-CNN-LSTM
Người thực hiện: Nhóm 1
Ngày báo cáo: 13/04/2026

1. Mục tiêu và Tổng quan
Tiếp nối thành công trong việc khép kín quy trình (End-to-End Pipeline) của tuần trước, mục tiêu trọng tâm của tuần này là tạo ra sự đột phá về mặt logic dự báo và tối ưu hóa hệ thống. Thay vì dự báo giá trị tuyệt đối (Giá đóng cửa), nhóm đã quyết định chuyển đổi mô hình sang dự báo Tỷ suất sinh lời (Returns - % thay đổi giá). Sự thay đổi mang tính nền tảng này giúp mô hình nắm bắt bản chất biến động của thị trường tốt hơn. Đồng thời, tuần này cũng tập trung giải quyết các lỗi phần cứng nghiêm trọng liên quan đến bộ nhớ GPU và hoàn thiện giao diện người dùng (UI/UX) đạt chuẩn chuyên nghiệp.

2. Các công việc đã thực hiện

Về lõi AI và Logic dự báo: Hoàn tất việc thay đổi cấu trúc dữ liệu mục tiêu (Target Variable) từ giá USD sang phần trăm biến động. Cấu trúc mạng học sâu đã được hoàn thiện với sự kết hợp của CNN (trích xuất đặc trưng cục bộ), LSTM (học chuỗi thời gian) và cơ chế Attention (tập trung vào các khung thời gian có biến động mạnh như kế hoạch đã đề ra). Thuật toán tối ưu GA-WOA đã được chạy lại trên tập dữ liệu mới, tìm ra cấu trúc gen xuất sắc với điểm thích nghi (Fitness) lên tới 0.9050 và sai số RMSE được kiểm soát ở mức 0.02 (tương đương 2%).

Về triển khai Backend: Tái cấu trúc file app.py để tương thích với logic mới. Hệ thống hiện tại sẽ giải mã (inverse transform) phần trăm dự báo, sau đó nhân với giá đóng cửa thực tế của phiên trước đó để nội suy ra giá trị USD dự báo cuối cùng, đảm bảo tính logic và loại bỏ hiện tượng "mô hình học vẹt/trễ pha".

Về giao diện (Frontend): Cập nhật mạnh mẽ trải nghiệm người dùng. Giao diện hiện tại không chỉ hiển thị mức giá dự báo mà còn cung cấp đối chiếu với giá đóng cửa gần nhất và tỷ lệ phần trăm tăng/giảm. Hệ thống tích hợp cơ chế nhận diện tự động, đổi màu kết quả động (Xanh cho xu hướng tăng, Đỏ cho xu hướng giảm) và tối ưu hóa thao tác người dùng (hỗ trợ phím Enter để kích hoạt luồng dự báo).

3. Các vấn đề đã gặp và cách khắc phục
Trong quá trình chuyển đổi sang dự báo Tỷ suất sinh lời, dự án đã đối mặt với một sự cố kỹ thuật nghiêm trọng ở cấp độ phần cứng: lỗi RuntimeError: CUDA error: an illegal instruction was encountered. Lỗi này làm treo hoàn toàn quá trình tính toán trên GPU.

Nguyên nhân: Khi chuyển sang tính toán % biến động, sự xuất hiện của các giá trị ngoại lai hoặc dữ liệu NaN/Inf (do chia cho 0 trong hàm pct_change) đã tạo ra các phép toán không hợp lệ, làm hỏng luồng lệnh (instruction set) của nhân CUDA. Ngoài ra, việc giữ nguyên Learning Rate lớn cho một tập dữ liệu có biên độ nhỏ (%) đã gây vỡ Gradient.

Khắc phục: Nhóm đã can thiệp vào file chuẩn bị dữ liệu (data_prep.py), thêm các bước làm sạch nghiêm ngặt (replace inf, dropna). Đồng thời, thông số không gian tìm kiếm của Thuật toán Di truyền (GA) được điều chỉnh để ép Learning Rate xuống mức rất nhỏ (0.0001). Kết quả, lỗi CUDA đã được xử lý triệt để, hệ thống chạy mượt mà trên môi trường GPU.

4. Kết quả đạt được và Kế hoạch tuần tới
Kết quả: Cấu trúc lõi của hệ thống đã hoàn thiện 100% với kiến trúc GA-WOA tối ưu hóa cho mạng CNN-LSTM-Attention. Ứng dụng Web đang chạy ổn định, giao diện trực quan, chuyên nghiệp và có khả năng đưa ra các nhận định thị trường bám rất sát thực tế nhờ logic tính toán dựa trên Return thay vì Price.

Kế hoạch tuần tới: Vì kiến trúc cốt lõi đã hoàn thiện và hoạt động trơn tru, định hướng trong tuần tới sẽ là:

Kiểm thử ngược (Backtesting): Xây dựng một module giả lập giao dịch để tính toán tỷ suất lợi nhuận thực tế (P&L) nếu đầu tư theo các tín hiệu dự báo của mô hình trong khoảng thời gian 1 năm qua.

Chuẩn bị triển khai thực tế (Deployment): Bắt đầu nghiên cứu việc đóng gói dự án (Docker) và đưa hệ thống từ môi trường cục bộ (localhost) lên một máy chủ đám mây (Cloud Server như Render, AWS, hoặc Heroku) để có thể truy cập từ xa.