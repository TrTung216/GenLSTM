
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. KIẾN TRÚC MẠNG — giữ nguyên từ bản gốc
# ==========================================

class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size=50, dropout_rate=0.2, cnn_filters=16, num_layers=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.relu   = nn.ReLU()
        self.lstm   = nn.LSTM(
            cnn_filters,
            hidden_layer_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.attention = nn.Linear(hidden_layer_size, 1)
        self.dropout   = nn.Dropout(dropout_rate)
        self.linear    = nn.Linear(hidden_layer_size, 1)

    def forward(self, input_seq):
        x = input_seq.permute(0, 2, 1)
        x = self.relu(self.conv1d(x))
        x = x.permute(0, 2, 1)
        lstm_out, _  = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vec  = torch.sum(attn_weights * lstm_out, dim=1)
        return self.linear(self.dropout(context_vec))


# ==========================================
# 2. CÁC HÀM TÍNH THÀNH PHẦN FITNESS
# ==========================================

def compute_rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return 1.0 / (1.0 + rmse)


def compute_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Thành phần 2: Directional Accuracy — % dự đoán đúng chiều tăng/giảm.

    Cơ chế:
      - Tính diff của y_true và y_pred (so ngày liền kề)
      - Nếu sign(diff_true) == sign(diff_pred) → dự đoán đúng chiều
      - DA = số lần đúng / tổng số lần

    Khoảng giá trị: [0, 1] — random guess ≈ 0.5, tốt > 0.6

    Tại sao quan trọng?
      Một mô hình có RMSE thấp nhưng DA thấp vẫn vô dụng với trader:
      dự báo giá $150.1 khi thực tế $150.3 (sai $0.2) nhưng chiều đúng → có lãi.
      Dự báo $149.9 (sai $0.4) chiều ngược → thua lỗ.
    """
    if len(y_true) < 2:
        return 0.5  # không đủ dữ liệu, trả về random baseline

    # Flatten về 1D nếu cần
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))

    # Loại bỏ các điểm không thay đổi (diff = 0) để tránh nhiễu
    mask = true_direction != 0
    if mask.sum() == 0:
        return 0.5

    correct = (true_direction[mask] == pred_direction[mask]).sum()
    return float(correct / mask.sum())


def compute_drawdown_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Thành phần 3: Drawdown Score — phạt khi mô hình sai chiều liên tiếp.

    Cơ chế:
      - Tạo chuỗi PnL giả định: +1 nếu đúng chiều, -1 nếu sai chiều
      - Tính cumulative PnL curve
      - Max Drawdown = mức sụt giảm lớn nhất từ đỉnh → đáy của curve
      - drawdown_score = 1 - max_drawdown (normalize về [0,1])

    Khoảng giá trị: [0, 1] — max_drawdown = 0 → score = 1 (hoàn hảo)

    Tại sao cần?
      DA chỉ đo trung bình đúng/sai, không phân biệt:
        Mô hình A: sai rải rác (ít nguy hiểm)
        Mô hình B: sai 10 lần liên tiếp (cháy tài khoản)
      Cả hai có thể cùng DA = 50%, nhưng B nguy hiểm hơn nhiều.
      Drawdown score phân biệt được điều này.
    """
    if len(y_true) < 2:
        return 1.0

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))

    # PnL giả định: +1 đúng chiều, -1 sai chiều, 0 bỏ qua flat
    pnl = np.where(true_dir == 0, 0,
          np.where(true_dir == pred_dir, 1, -1))

    # Cumulative PnL curve
    cum_pnl = np.cumsum(pnl)

    # Max drawdown: mức giảm lớn nhất từ đỉnh tích lũy
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns   = running_max - cum_pnl
    max_dd      = drawdowns.max()

    # Normalize: chia cho số bước để về [0, 1]
    max_possible_dd = len(pnl)  # worst case: sai hết
    normalized_dd   = max_dd / max_possible_dd if max_possible_dd > 0 else 0

    return float(1.0 - normalized_dd)


# ==========================================
# 3. HÀM FITNESS TỔNG HỢP
# ==========================================

# Trọng số mặc định — tổng = 1.0
# Điều chỉnh theo mục tiêu:
#   Nghiên cứu học thuật : W_RMSE=0.5, W_DIR=0.3, W_DD=0.2
#   Ứng dụng giao dịch   : W_RMSE=0.3, W_DIR=0.5, W_DD=0.2
#   Quản trị rủi ro      : W_RMSE=0.3, W_DIR=0.3, W_DD=0.4
W_RMSE      = 0.40
W_DIRECTION = 0.40
W_DRAWDOWN  = 0.20


def combined_fitness(
    y_true           : np.ndarray,
    y_pred           : np.ndarray,
    w_rmse           : float = W_RMSE,
    w_direction      : float = W_DIRECTION,
    w_drawdown       : float = W_DRAWDOWN,
    verbose          : bool  = False,
    return_components: bool  = False,
):
    assert abs(w_rmse + w_direction + w_drawdown - 1.0) < 1e-6, \
        "Tổng trọng số phải = 1.0"

    rmse_score = compute_rmse_score(y_true, y_pred)
    da_score   = compute_directional_accuracy(y_true, y_pred)
    dd_score   = compute_drawdown_score(y_true, y_pred)

    fitness = w_rmse * rmse_score + w_direction * da_score + w_drawdown * dd_score

    if verbose:
        rmse_val = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
        print(f"  RMSE          : {rmse_val:.6f}  → score = {rmse_score:.4f}  (×{w_rmse})")
        print(f"  Directional   : {da_score*100:.1f}%         → score = {da_score:.4f}  (×{w_direction})")
        print(f"  Drawdown      : {(1-dd_score)*100:.1f}% dd   → score = {dd_score:.4f}  (×{w_drawdown})")
        print(f"  ─────────────────────────────────────────")
        print(f"  Fitness Total : {fitness:.6f}")

    if return_components:
        return fitness, {"rmse_score": rmse_score, "da_score": da_score, "dd_score": dd_score}
    return fitness


# ==========================================
# 4. HÀM EVALUATE_FITNESS CHO GA (THAY THẾ BẢN GỐC)
# ==========================================

def evaluate_fitness(
    chromosome,
    X_train, y_train,
    X_val,   y_val,
    w_rmse           : float = W_RMSE,
    w_direction      : float = W_DIRECTION,
    w_drawdown       : float = W_DRAWDOWN,
    verbose          : bool  = False,
    return_components: bool  = False,
):
    units, dropout_rate, lr, batch_size, window_size, cnn_filters, num_layers = chromosome

    units       = int(units)
    batch_size  = int(batch_size)
    cnn_filters = int(cnn_filters)
    num_layers  = int(num_layers)

    # ── DataLoader ───────────────────────────────────────────
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(torch.cuda.is_available()),
    )

    # ── Khởi tạo model ───────────────────────────────────────
    model = CNN_LSTM(
        input_size        = X_train.shape[2],
        hidden_layer_size = units,
        dropout_rate      = dropout_rate,
        cnn_filters       = cnn_filters,
        num_layers        = num_layers,
    ).to(device)

    loss_fn   = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ── Huấn luyện ngắn với Early Stopping ───────────────────
    best_val_loss    = float("inf")
    patience_counter = 0
    PATIENCE         = 5
    MAX_EPOCHS       = 25

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    for epoch in range(MAX_EPOCHS):
        model.train()
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(seq), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            if verbose:
                print(f"  Early stop tại epoch {epoch+1}")
            break

    # ── Lấy dự báo trên validation ───────────────────────────
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t).cpu().numpy()

    y_val_np = y_val.flatten() if hasattr(y_val, 'flatten') else np.array(y_val).flatten()

    # ── Tính fitness kết hợp ─────────────────────────────────
    return combined_fitness(
        y_true           = y_val_np,
        y_pred           = val_preds,
        w_rmse           = w_rmse,
        w_direction      = w_direction,
        w_drawdown       = w_drawdown,
        verbose          = verbose,
        return_components= return_components,
    )


# ==========================================
# 5. HELPER: so sánh fitness cũ vs mới
# ==========================================

def evaluate_fitness_legacy(chromosome, X_train, y_train, X_val, y_val):
    """
    Giữ lại bản gốc để so sánh trong quá trình chuyển đổi.
    Xóa sau khi xác nhận bản mới hoạt động ổn định.
    """
    units, dropout_rate, lr, batch_size, window_size, cnn_filters, num_layers = chromosome
    batch_size  = int(batch_size)
    units       = int(units)
    cnn_filters = int(cnn_filters)
    num_layers  = int(num_layers)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size, shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    model = CNN_LSTM(X_train.shape[2], units, dropout_rate, cnn_filters, num_layers).to(device)
    loss_fn   = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    best_val = float("inf"); p = 0
    for _ in range(25):
        model.train()
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            loss_fn(model(seq), labels).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(X_val_t), y_val_t).item()
        if v < best_val: best_val = v; p = 0
        else: p += 1
        if p >= 5: break

    model.eval()
    with torch.no_grad():
        preds = model(X_val_t).cpu().numpy()

    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_val, preds)
    r2  = max(r2_score(y_val, preds), 0)
    return 1.0 / (mse + 0.1 * (1 - r2) + 1e-7)


def compare_fitness_functions(chromosome, X_train, y_train, X_val, y_val):
    print("=" * 50)
    print("  FITNESS FUNCTION COMPARISON")
    print("=" * 50)

    print("\n[Legacy] MSE + R2:")
    legacy = evaluate_fitness_legacy(chromosome, X_train, y_train, X_val, y_val)
    print(f"  Fitness = {legacy:.6f}")

    print("\n[New] RMSE + Directional + Drawdown:")
    new = evaluate_fitness(chromosome, X_train, y_train, X_val, y_val, verbose=True)

    print(f"\n  Legacy : {legacy:.6f}")
    print(f"  New    : {new:.6f}")
    print("=" * 50)
    return {"legacy": legacy, "new": new}