BÁO CÁO TIẾN ĐỘ TUẦN
Dự án: Xây dựng Hệ thống Dự báo Giá Cổ phiếu bằng GA-CNN-LSTM
Người thực hiện: Nhóm 1
Ngày báo cáo: 06/04/2026

1. Mục tiêu và Tổng quan
Trong tuần qua, mục tiêu trọng tâm của dự án là hoàn thiện vòng lặp khép kín (End-to-End Pipeline) cho hệ thống, từ khâu thu thập dữ liệu, huấn luyện mô hình, cho đến khi triển khai thực tế trên ứng dụng Web. Trọng tâm công việc được đặt vào việc xử lý triệt để các lỗi đồng bộ dữ liệu giữa môi trường huấn luyện và môi trường thực tế, đồng thời đảm bảo giao diện người dùng hoạt động mượt mà, trả về kết quả dự báo chính xác cho phiên giao dịch tiếp theo.

2. Các công việc đã thực hiện
Về phần cốt lõi trí tuệ nhân tạo, Thuật toán Di truyền (GA) đã được nâng cấp và mở rộng không gian tìm kiếm lên 6 tham số (bao gồm Units, Dropout, Learning Rate, Batch Size, Window Size và CNN Filters). Điều này giúp hệ thống tự động tìm ra cấu trúc mạng tối ưu thay vì thiết lập thủ công. Dữ liệu đầu vào được nâng cấp thông qua việc tích hợp 14 chỉ báo kỹ thuật quan trọng như SMA, EMA, RSI, MACD. Đặc biệt, vấn đề sai lệch giá dự báo đã được giải quyết hoàn toàn nhờ việc tách biệt và xuất khẩu thành công các bộ chuẩn hóa dữ liệu (Scaler) cho các tập đặc trưng và mục tiêu.

Về phần triển khai hệ thống (Backend & Frontend), kiến trúc mạng PyTorch đã được tinh chỉnh để đồng bộ hóa hoàn toàn với cấu trúc mô hình tối ưu. Logic của hệ thống dự báo cũng được nâng cấp với thuật toán nhận diện và tính toán chính xác ngày giao dịch tiếp theo của thị trường (bỏ qua các ngày cuối tuần). Cùng với đó, giao diện người dùng được cấu trúc lại, khắc phục lỗi bất đồng bộ của JavaScript và tích hợp cơ chế xử lý lỗi từ máy chủ để hiển thị kết quả trực quan hơn.

3. Các vấn đề đã gặp và cách khắc phục
Trong quá trình triển khai, dự án đã gặp phải một số vướng mắc kỹ thuật. Nổi bật là lỗi xung đột luồng xử lý đồ họa (Runtime Error) khi hệ thống cố gắng vẽ và lưu biểu đồ, lỗi từ chối tải trọng số mô hình do khác biệt số lượng nơ-ron lớp ẩn giữa mô hình được GA sinh ra và mô hình khởi tạo trên Web, cũng như các lỗi mất kết nối máy chủ cục bộ do sai sót trong cú pháp DOM của giao diện. Toàn bộ các vấn đề này đã được rà soát và xử lý dứt điểm bằng cách quản lý lại tài nguyên bộ nhớ đồ họa, đồng bộ hóa cấu hình cứng của mô hình và viết lại đoạn mã giao tiếp API.

4. Kết quả đạt được và Kế hoạch tuần tới
Đến thời điểm hiện tại, hệ thống đã vận hành thông suốt 100%. Nền tảng cho phép tự động tải dữ liệu mới nhất của bất kỳ mã chứng khoán nào, thực hiện các phép tính kỹ thuật và đưa ra dự báo giá đóng cửa cho phiên làm việc tiếp theo của thị trường.

Trong tuần tới, hướng phát triển của dự án sẽ tập trung vào việc tối ưu hóa hiệu suất của mô hình. Các phương án dự kiến triển khai bao gồm: nghiên cứu tích hợp cơ chế Attention để giúp mạng LSTM nhận diện các khung thời gian có biến động giá quan trọng, thử nghiệm hàm mất mát HuberLoss nhằm hạn chế sự nhạy cảm của mô hình với các mức giá ngoại lai, và áp dụng bộ chuẩn hóa RobustScaler để xử lý nhiễu dữ liệu thị trường một cách hiệu quả hơn.