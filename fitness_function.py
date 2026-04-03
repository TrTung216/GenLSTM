import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Tự động nhận diện GPU (CUDA) nếu có, ngược lại dùng CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. ĐỊNH NGHĨA KIẾN TRÚC MẠNG LAI CNN-LSTM
class CNN_LSTM(nn.Module):
    def __init__(self, input_size=7, hidden_layer_size=50, dropout_rate=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        
        # 1.1 Lớp 1D-CNN (Trích xuất đặc trưng nến cục bộ)
        # in_channels = số lượng biến đầu vào (7)
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        # 1.2 Lớp LSTM (Học tính tuần tự thời gian)
        # Đầu vào của LSTM giờ là 16 (tương ứng với out_channels của lớp CNN)
        self.lstm = nn.LSTM(16, hidden_layer_size, batch_first=True)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_layer_size, 1)

    def forward(self, input_seq):
        # input_seq: (batch_size, seq_length, input_size)
        # Đảo chiều Tensor để phù hợp với Conv1d của PyTorch: (batch_size, input_size, seq_length)
        x = input_seq.permute(0, 2, 1)
        
        # Đi qua CNN
        x = self.conv1d(x)
        x = self.relu(x)
        
        # Đảo chiều lại cho LSTM: (batch_size, seq_length, 16)
        x = x.permute(0, 2, 1)
        
        # Đi qua LSTM
        lstm_out, _ = self.lstm(x)
        
        # Chỉ lấy đầu ra của bước thời gian cuối cùng để dự báo
        predictions = self.linear(self.dropout(lstm_out[:, -1, :]))
        return predictions

# 2. HÀM TÍNH FITNESS CHO GA
def evaluate_fitness(chromosome, X_train, y_train, X_val, y_val):
    # Giải mã gene (Không gian tìm kiếm: Số nơ-ron, Tỷ lệ Dropout, Learning Rate, Batch Size)
    units, dropout_rate, lr, batch_size = chromosome
    
    # Ép kiểu an toàn cho batch_size và units thành số nguyên
    batch_size = int(batch_size)
    units = int(units)

    # Chuyển dữ liệu Numpy sang PyTorch Tensor và đẩy lên GPU
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    # Tạo DataLoader để chia lô (batching)
    train_data = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    # Khởi tạo mô hình CNN-LSTM với 7 biến đầu vào
    model = CNN_LSTM(input_size=7, hidden_layer_size=units, dropout_rate=dropout_rate).to(device)
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
            optimizer.zero_grad() 
            y_pred = model(seq)   
            loss = loss_function(y_pred, labels) 
            loss.backward()       
            optimizer.step()      

        # Đánh giá trên tập Validation
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
            break # Dừng sớm

    # --- TÍNH TOÁN FITNESS SCORE MỚI (CHUẨN 2025) ---
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val_t).cpu().numpy()

    # Tính MSE
    mse = mean_squared_error(y_val, val_predictions)
    
    # Tính R-squared
    r2 = r2_score(y_val, val_predictions)
    
    # Xử lý R2: Nếu dự báo quá tệ (R2 < 0), ta quy nó về 0 để hàm phạt không bị nhiễu
    r2_clipped = max(r2, 0)
    
    # Hàm mất mát lai: MSE + (1 - R^2). Trọng số 0.1 giúp cân bằng 2 đại lượng.
    custom_loss = mse + 0.1 * (1 - r2_clipped)
    
    # Thuật toán GA tìm kiếm giá trị MAX, nên Fitness tỷ lệ nghịch với Loss
    fitness_score = 1.0 / (custom_loss + 1e-7)
    
    return fitness_score