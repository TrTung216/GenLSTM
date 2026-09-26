
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

try:
    from src.model import CNN_LSTM
except ModuleNotFoundError:
    # Hỗ trợ chạy trực tiếp: python src/ga_lstm.py
    from model import CNN_LSTM

try:
    from src.training import train_with_early_stopping
    from src.validation import walk_forward_splits
except ModuleNotFoundError:
    from training import train_with_early_stopping
    from validation import walk_forward_splits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. CÁC HÀM TÍNH THÀNH PHẦN FITNESS
# ==========================================

def compute_rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return 1.0 / (1.0 + rmse)


def compute_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the fraction of correctly predicted return directions.

    The model target is daily return (Close.pct_change()), therefore direction
    is the sign of each return itself:
      positive return -> price up
      negative return -> price down

    Comparing np.diff(y_true) and np.diff(y_pred) would instead measure whether
    the return increased/decreased relative to the previous return, which is a
    different quantity.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if len(y_true) == 0 or len(y_true) != len(y_pred):
        return 0.0

    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)

    # Ignore exactly-flat true returns because they have no up/down direction.
    mask = true_direction != 0
    if mask.sum() == 0:
        return 0.5

    correct = (true_direction[mask] == pred_direction[mask]).sum()
    return float(correct / mask.sum())


def compute_drawdown_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Penalize long sequences of wrong predicted return directions.

    Since y_true/y_pred are daily returns, trading direction is determined by
    the sign of each return rather than the difference between adjacent returns.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if len(y_true) == 0 or len(y_true) != len(y_pred):
        return 1.0

    true_dir = np.sign(y_true)
    pred_dir = np.sign(y_pred)

    # +1 for correct direction, -1 for wrong direction, 0 for flat true return.
    pnl = np.where(
        true_dir == 0,
        0,
        np.where(true_dir == pred_dir, 1, -1),
    )

    cum_pnl = np.cumsum(pnl)
    running_max = np.maximum.accumulate(np.concatenate(([0], cum_pnl)))[1:]
    drawdowns = running_max - cum_pnl
    max_dd = drawdowns.max() if len(drawdowns) else 0

    max_possible_dd = len(pnl)
    normalized_dd = max_dd / max_possible_dd if max_possible_dd > 0 else 0
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
        print("  ─────────────────────────────────────────")
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

    # ── Khởi tạo model ───────────────────────────────────────
    model = CNN_LSTM(
        input_size        = X_train.shape[2],
        hidden_layer_size = units,
        dropout_rate      = dropout_rate,
        cnn_filters       = cnn_filters,
        num_layers        = num_layers,
    ).to(device)

    result = train_with_early_stopping(
        model, X_train, y_train, X_val, y_val, lr, batch_size, device,
        max_epochs=25, patience=5,
    )
    model = result.model

    # ── Lấy dự báo trên validation ───────────────────────────
    model.eval()
    with torch.no_grad():
        val_preds = model(torch.as_tensor(X_val, dtype=torch.float32, device=device)).cpu().numpy()

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


def evaluate_walk_forward_fitness(
    chromosome,
    X_development,
    y_development,
    n_splits=3,
    **fitness_kwargs,
):
    """Average chromosome fitness over expanding validation folds.

    ``X_development`` must exclude the final test partition.  A new model is
    trained for every fold, preventing weights from leaking across folds.
    """
    fold_results = []
    for train_idx, validation_idx in walk_forward_splits(len(X_development), n_splits):
        score, components = evaluate_fitness(
            chromosome,
            X_development[train_idx], y_development[train_idx],
            X_development[validation_idx], y_development[validation_idx],
            return_components=True,
            **fitness_kwargs,
        )
        fold_results.append((score, components))
    mean_components = {
        key: float(np.mean([components[key] for _, components in fold_results]))
        for key in fold_results[0][1]
    }
    mean_components["fold_fitness_std"] = float(np.std([score for score, _ in fold_results]))
    return float(np.mean([score for score, _ in fold_results])), mean_components


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

    best_val = float("inf")
    p = 0
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
        if v < best_val:
            best_val = v
            p = 0
        else:
            p += 1
        if p >= 5:
            break

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
