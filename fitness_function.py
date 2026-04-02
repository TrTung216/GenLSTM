import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error

# Tự động nhận diện GPU (CUDA) nếu có, ngược lại dùng CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. ĐỊNH NGHĨA KIẾN TRÚC MẠNG LSTM BẰNG PYTORCH
class PyTorchLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_layer_size=50, dropout_rate=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        
        # 1. Thêm tham số bidirectional=True vào mạng LSTM
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True, bidirectional=True)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # 2. Quan trọng: Khi dùng Bi-LSTM, đầu ra sẽ bị nhân đôi. 
        # Nên lớp Linear cuối cùng phải nhận đầu vào là hidden_layer_size * 2
        self.linear = nn.Linear(hidden_layer_size * 2, 1)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        # Chỉ lấy đầu ra của bước thời gian cuối cùng để dự báo
        predictions = self.linear(self.dropout(lstm_out[:, -1, :]))
        return predictions

# 2. HÀM TÍNH FITNESS CHO GA
def evaluate_fitness(chromosome, X_train, y_train, X_val, y_val):
    # Giải mã gene
    units, dropout_rate, lr, batch_size = chromosome

    # Chuyển dữ liệu Numpy sang PyTorch Tensor và đẩy lên GPU
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    # Tạo DataLoader để chia lô (batching)
    train_data = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    # Khởi tạo mô hình, hàm mất mát (MSE) và bộ tối ưu (Adam)
    model = PyTorchLSTM(input_size=4, hidden_layer_size=units, dropout_rate=dropout_rate).to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Cấu hình Early Stopping
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    # Vòng lặp huấn luyện (Epochs)
    for epoch in range(30):
        model.train()
        for seq, labels in train_loader:
            optimizer.zero_grad() # Xóa gradient cũ
            y_pred = model(seq)   # Dự đoán
            loss = loss_function(y_pred, labels) # Tính sai số
            loss.backward()       # Lan truyền ngược
            optimizer.step()      # Cập nhật trọng số

        # Đánh giá trên tập Validation để kích hoạt Early Stopping
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = loss_function(val_preds, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break # Dừng sớm nếu không cải thiện

    # Dự đoán lần cuối trên Validation để lấy RMSE tính Fitness
    model.eval()
    with torch.no_grad():
        # Đẩy dữ liệu về lại CPU và chuyển thành Numpy để dùng với sklearn
        val_predictions = model(X_val_t).cpu().numpy()

    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    fitness_score = 1.0 / (rmse + 1e-7)
    return fitness_score