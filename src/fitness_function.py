import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Tự động nhận diện GPU (CUDA) nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. ĐỊNH NGHĨA KIẾN TRÚC MẠNG LAI CNN-LSTM (NÂNG CẤP STACKED LSTM)
# ==========================================
class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size=50, dropout_rate=0.2, cnn_filters=16, num_layers=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        
        # Lớp CNN trích xuất đặc trưng không gian
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        # Lớp LSTM nâng cấp lên Stacked LSTM (nhiều lớp chồng nhau)
        # dropout chỉ có tác dụng nếu num_layers > 1
        self.lstm = nn.LSTM(
            cnn_filters, 
            hidden_layer_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Lõi Attention
        self.attention = nn.Linear(hidden_layer_size, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_layer_size, 1)

    def forward(self, input_seq):
        # CNN layer
        x = input_seq.permute(0, 2, 1) # [batch, features, seq_len]
        x = self.relu(self.conv1d(x))
        x = x.permute(0, 2, 1) # [batch, seq_len, filters]
        
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Attention Mechanism
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Final prediction
        predictions = self.linear(self.dropout(context_vector))
        return predictions

# ==========================================
# 2. HÀM TÍNH FITNESS CHO GA (7 GENES)
# ==========================================
def evaluate_fitness(chromosome, X_train, y_train, X_val, y_val):
    units, dropout_rate, lr, batch_size, window_size, cnn_filters, num_layers = chromosome
    
    # Ép kiểu
    batch_size = int(batch_size)
    units = int(units)
    cnn_filters = int(cnn_filters)
    num_layers = int(num_layers)

    # 1. Giữ Tensor ở CPU để dùng pin_memory
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    train_data = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 2. Khởi tạo mô hình với Gene num_layers
    model = CNN_LSTM(
        input_size=X_train.shape[2], 
        hidden_layer_size=units, 
        dropout_rate=dropout_rate,
        cnn_filters=cnn_filters,
        num_layers=num_layers
    ).to(device)
    
    loss_function = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3. Huấn luyện ngắn (GA evaluation)
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(25):
        model.train()
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad() 
            y_pred = model(seq)   
            loss = loss_function(y_pred, labels) 
            loss.backward()       
            optimizer.step()      

        # Đánh giá nhanh trên tập Validation
        model.eval()
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
        
        with torch.no_grad():
            val_preds = model(X_val_t)
            current_val_loss = loss_function(val_preds, y_val_t).item()

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # 4. Tính toán điểm Fitness
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val_t).cpu().numpy()

    mse = mean_squared_error(y_val, val_predictions)
    r2 = r2_score(y_val, val_predictions)
    r2_clipped = max(r2, 0)
    
    # Kết hợp MSE và R2 để đảm bảo xu hướng dự báo đúng
    custom_loss = mse + 0.1 * (1 - r2_clipped)
    return 1.0 / (custom_loss + 1e-7)