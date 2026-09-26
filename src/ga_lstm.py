"""
ga_lstm.py — GA-WOA Hybrid với Fitness Function kết hợp
=========================================================
Thay đổi so với bản gốc:
  - evaluate_fitness() mới dùng 3 tiêu chí: RMSE + Directional Accuracy + Drawdown
  - Thêm FITNESS_WEIGHTS config ở đầu file để dễ điều chỉnh
  - Thêm log chi tiết từng thành phần fitness khi verbose=True
  - Phần còn lại (GA-WOA loop, crossover, mutate, WOA) giữ nguyên 100%
"""

import numpy as np
import random
import copy
import math
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import yfinance as yf

try:
    from src.data_prep import prepare_train_validation_test, fetch_macro_data
    from src.fitness_function import (
        evaluate_walk_forward_fitness, CNN_LSTM, device,
        compute_directional_accuracy,
    )
    from src.training import train_with_early_stopping
except ModuleNotFoundError:
    from data_prep import prepare_train_validation_test, fetch_macro_data
    from fitness_function import (
        evaluate_walk_forward_fitness, CNN_LSTM, device,
        compute_directional_accuracy,
    )
    from training import train_with_early_stopping

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Current Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ==========================================
# CONFIG
# ==========================================

TICKER_SYMBOL = 'AAPL'
START_DATE    = '2015-01-01'
# Tổng phải = 1.0
# Gợi ý:
#   Nghiên cứu học thuật : W_RMSE=0.5, W_DIR=0.3, W_DD=0.2
#   Ứng dụng giao dịch   : W_RMSE=0.3, W_DIR=0.5, W_DD=0.2
#   Quản trị rủi ro      : W_RMSE=0.3, W_DIR=0.3, W_DD=0.4
FITNESS_WEIGHTS = {
    "w_rmse"     : 0.40,
    "w_direction": 0.40,
    "w_drawdown" : 0.20,
}

# ==========================================
# 1. NGÀY GIAO DỊCH CUỐI TỪ NYSE
# ==========================================

nyse     = mcal.get_calendar('NYSE')
now      = datetime.now()
schedule = nyse.schedule(
    start_date=(now - timedelta(days=10)).strftime('%Y-%m-%d'),
    end_date=now.strftime('%Y-%m-%d')
)
END_DATE = schedule.index[-1].strftime('%Y-%m-%d')
print(f"Ngày giao dịch cuối cùng xác định bởi NYSE: {END_DATE}")

# ==========================================
# 2. KHÔNG GIAN TÌM KIẾM (7 GENES) — giữ nguyên
# ==========================================

SPACE_UNITS   = [32, 64, 96, 128]
SPACE_DROPOUT = [0.05, 0.1, 0.15]
SPACE_LR      = [0.0001, 0.0005, 0.001, 0.005]
SPACE_BATCH   = [16, 32, 64]
SPACE_WINDOW  = [20, 30, 45, 60]
SPACE_FILTERS = [16, 32, 64]
SPACE_LAYERS  = [1, 2]

SPACES_DICT = {
    'SPACE_UNITS'  : SPACE_UNITS,
    'SPACE_DROPOUT': SPACE_DROPOUT,
    'SPACE_LR'     : SPACE_LR,
    'SPACE_BATCH'  : SPACE_BATCH,
    'SPACE_WINDOW' : SPACE_WINDOW,
    'SPACE_FILTERS': SPACE_FILTERS,
    'SPACE_LAYERS' : SPACE_LAYERS,
}

POPULATION_SIZE = 20
GENERATIONS     = 15
CROSSOVER_RATE  = 0.8
TOURNAMENT_SIZE = 3

# ==========================================
# 3. GA OPERATORS — giữ nguyên hoàn toàn
# ==========================================

def create_individual():
    return [
        random.choice(SPACE_UNITS),
        random.choice(SPACE_DROPOUT),
        random.choice(SPACE_LR),
        random.choice(SPACE_BATCH),
        random.choice(SPACE_WINDOW),
        random.choice(SPACE_FILTERS),
        random.choice(SPACE_LAYERS),
    ]

def tournament_selection(population, fitness_scores):
    selected = []
    for _ in range(len(population)):
        idxs    = random.sample(range(len(population)), TOURNAMENT_SIZE)
        best_i  = idxs[0]
        for idx in idxs[1:]:
            if fitness_scores[idx] > fitness_scores[best_i]:
                best_i = idx
        selected.append(copy.deepcopy(population[best_i]))
    return selected

def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, 6)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    return parent1, parent2

def mutate(individual, mut_rate):
    if random.random() < mut_rate:
        gene_idx = random.randint(0, 6)
        choices  = [SPACE_UNITS, SPACE_DROPOUT, SPACE_LR, SPACE_BATCH,
                    SPACE_WINDOW, SPACE_FILTERS, SPACE_LAYERS]
        individual[gene_idx] = random.choice(choices[gene_idx])
    return individual

def woa_refinement(chromosome, best_chromosome, current_gen, max_gen, spaces):
    new_chrom = list(chromosome)
    a = 2 - current_gen * (2 / max_gen)

    for i in range(len(chromosome)):
        p = random.random()
        if p < 0.5:
            r     = random.random()
            A     = 2 * a * r - a
            C     = 2 * r
            D     = abs(C * best_chromosome[i] - chromosome[i])
            new_val = best_chromosome[i] - A * D
        else:
            D_prime = abs(best_chromosome[i] - chromosome[i])
            b       = 1.0
            spiral = random.uniform(-1, 1)
            new_val = (D_prime * math.exp(b * spiral)
                       * math.cos(2 * math.pi * spiral) + best_chromosome[i])

        if i == 1:   # Dropout — continuous
            new_chrom[i] = max(0.0, min(0.5, round(new_val, 2)))
        elif i == 2: # LR — snap về closest trong space
            new_chrom[i] = min(spaces['SPACE_LR'], key=lambda x: abs(x - new_val))
        else:        # Gene rời rạc — snap về closest
            mapping = {
                0: spaces['SPACE_UNITS'],
                3: spaces['SPACE_BATCH'],
                4: spaces['SPACE_WINDOW'],
                5: spaces['SPACE_FILTERS'],
                6: spaces['SPACE_LAYERS'],
            }
            if i in mapping:
                new_chrom[i] = min(mapping[i], key=lambda x: abs(x - new_val))

    return new_chrom

# ==========================================
# 4. VÒNG LẶP TIẾN HÓA GA-WOA
# ==========================================

def run_ga_lstm(df_raw):
    print(f"Bắt đầu GA-WOA với {len(df_raw)} dòng dữ liệu.")
    print(f"Fitness weights: {FITNESS_WEIGHTS}")   # [MỚI] log config

    population              = [create_individual() for _ in range(POPULATION_SIZE)]
    best_chromosome_overall = None
    best_fitness_overall    = -1
    history_best_fitness    = []
    generation_stats        = []

    # [MỚI] Lưu lịch sử từng thành phần fitness để vẽ sau
    history_components = []   # list of dict {rmse, da, dd} mỗi gen

    for gen in range(GENERATIONS):
        print(f"\n{'='*55}")
        print(f"  Thế hệ {gen + 1}/{GENERATIONS}")
        print(f"{'='*55}")

        fitness_scores = []
        gen_components = []   # [MỚI] components của gen này

        for i, chromosome in enumerate(population):
            # window_size = gene[4]; truyền ticker để tải VIX/TNX đúng mã
            # [FIX 1] df_raw đã có VIX/TNX từ trước — không cần truyền ticker
            data_package = prepare_train_validation_test(
                df_raw, chromosome[4], scaler_fit_fraction=0.5,
            )

            if data_package[0] is None or len(data_package[0]) == 0:
                fitness = 1e-6
                gen_components.append(None)
            else:
                X_train, y_train, X_val, y_val, _, _, _, _ = data_package
                X_development = np.concatenate((X_train, X_val))
                y_development = np.concatenate((y_train, y_val))

                if len(X_development) < 6:
                    fitness = 1e-6
                    gen_components.append(None)
                else:
                    fitness, components = evaluate_walk_forward_fitness(
                        chromosome,
                        X_development, y_development,
                        n_splits=3,
                        **FITNESS_WEIGHTS,
                    )
                    gen_components.append(components)

            fitness_scores.append(fitness)
            _log_individual(i, chromosome, fitness, gen_components[-1])

        # Sắp xếp theo fitness giảm dần
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population     = [population[idx] for idx in sorted_indices]
        fitness_scores = [fitness_scores[idx] for idx in sorted_indices]
        gen_components = [gen_components[idx] for idx in sorted_indices]

        if fitness_scores[0] > best_fitness_overall:
            best_fitness_overall    = fitness_scores[0]
            best_chromosome_overall = copy.deepcopy(population[0])

        history_best_fitness.append(best_fitness_overall)
        generation_stats.append({
            "generation": gen + 1,
            "best_fitness": float(fitness_scores[0]),
            "best_fitness_overall": float(best_fitness_overall),
            "mean_fitness": float(np.mean(fitness_scores)),
            "std_fitness": float(np.std(fitness_scores)),
            "unique_chromosomes": len({tuple(chromosome) for chromosome in population}),
            "population_size": len(population),
        })

        # [MỚI] Lưu components của cá thể tốt nhất gen này
        if gen_components[0] is not None:
            history_components.append(gen_components[0])

        print(f"\n  ★ Best overall: {best_chromosome_overall}")
        print(f"    Fitness = {best_fitness_overall:.4f}")

        # ── GA-WOA HYBRID — giữ nguyên logic ─────────────────────────────
        next_gen  = [copy.deepcopy(best_chromosome_overall)]   # Elitism

        num_woa = int(0.3 * POPULATION_SIZE)
        for i in range(1, num_woa):
            refined = woa_refinement(population[i], best_chromosome_overall,
                                     gen, GENERATIONS, SPACES_DICT)
            next_gen.append(refined)

        selected = tournament_selection(population, fitness_scores)
        num_ga   = POPULATION_SIZE - num_woa

        for i in range(0, num_ga, 2):
            p1 = selected[random.randint(0, len(selected) - 1)]
            p2 = selected[random.randint(0, len(selected) - 1)]
            c1, c2 = crossover(p1, p2)
            dynamic_mut_rate = random.uniform(0.1, 0.5)
            next_gen.append(mutate(c1, dynamic_mut_rate))
            if len(next_gen) < POPULATION_SIZE:
                next_gen.append(mutate(c2, dynamic_mut_rate))

        population = next_gen[:POPULATION_SIZE]

    os.makedirs("experiments", exist_ok=True)
    pd.DataFrame(generation_stats).to_csv("experiments/ga_generation_stats.csv", index=False)
    return best_chromosome_overall, history_best_fitness, history_components, generation_stats


def _log_individual(idx, chromosome, fitness, components):
    """In thông tin một cá thể — tách ra để run_ga_lstm gọn hơn."""
    if components:
        print(
            f"  [{idx+1:02d}] units={chromosome[0]:3d} lr={chromosome[2]:.4f} "
            f"win={chromosome[4]:2d} | "
            f"fit={fitness:.4f}  "
            f"(rmse={components['rmse_score']:.3f} "
            f"da={components['da_score']*100:.1f}% "
            f"dd={components['dd_score']:.3f})"
        )
    else:
        print(f"  [{idx+1:02d}] {chromosome} → fitness=1e-6 (data error)")


# ==========================================
# 5. VẼ ĐỒ THỊ HỘI TỤ — thêm subplot components
# ==========================================

def plot_convergence(history_best_fitness, history_components, generation_stats):
    """
    Vẽ 2 subplot:
      - Trên: đường hội tụ fitness tổng (như bản gốc)
      - Dưới: [MỚI] từng thành phần fitness qua các thế hệ
    """
    gens = range(1, len(history_best_fitness) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("GA-WOA Convergence", fontsize=14, fontweight='bold')

    # Subplot 1: Tổng fitness
    axes[0].plot(gens, history_best_fitness, 'b-o', linewidth=2, markersize=5)
    axes[0].set_xlabel("Thế hệ")
    axes[0].set_ylabel("Best Fitness")
    axes[0].set_title("Quá trình hội tụ Fitness tổng")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    # Subplot 2: Từng thành phần
    if history_components:
        comp_gens   = range(1, len(history_components) + 1)
        rmse_hist   = [c['rmse_score']    for c in history_components]
        da_hist     = [c['da_score']      for c in history_components]
        dd_hist     = [c['dd_score']      for c in history_components]

        axes[1].plot(comp_gens, rmse_hist, 'g-s', label=f"RMSE score (×{FITNESS_WEIGHTS['w_rmse']})",     linewidth=1.5)
        axes[1].plot(comp_gens, da_hist,   'r-^', label=f"Directional acc (×{FITNESS_WEIGHTS['w_direction']})", linewidth=1.5)
        axes[1].plot(comp_gens, dd_hist,   'b-o', label=f"Drawdown score (×{FITNESS_WEIGHTS['w_drawdown']})",  linewidth=1.5)
        axes[1].set_xlabel("Thế hệ")
        axes[1].set_ylabel("Score")
        axes[1].set_title("Breakdown từng thành phần Fitness")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)

    plt.tight_layout()
    os.makedirs("experiments/plots", exist_ok=True)
    plt.savefig("experiments/plots/ga_convergence.png", dpi=150)
    plt.close(fig)
    print("Đã lưu experiments/plots/ga_convergence.png")

    stats = pd.DataFrame(generation_stats)
    plots = {
        "ga_best_fitness.png": ("best_fitness_overall", "Best fitness"),
        "ga_mean_fitness.png": ("mean_fitness", "Mean fitness"),
        "ga_population_diversity.png": ("unique_chromosomes", "Unique chromosomes"),
    }
    for filename, (column, ylabel) in plots.items():
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.plot(stats["generation"], stats[column], marker="o")
        axis.set(xlabel="Generation", ylabel=ylabel, title=ylabel)
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(os.path.join("experiments/plots", filename), dpi=150)
        plt.close(figure)


# ==========================================
# 6. HUẤN LUYỆN CUỐI CÙNG & XUẤT FILE — giữ nguyên
# ==========================================

if __name__ == "__main__":
    print("--- Đang tải dữ liệu gốc từ Yahoo Finance ---")
    df_raw = yf.download(TICKER_SYMBOL, start=START_DATE, end=END_DATE, progress=False)

    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)

    if df_raw.empty or len(df_raw) < 100:
        print("Lỗi: Không có dữ liệu hoặc quá ngắn.")
    else:
        # [FIX 1] Tải VIX + TNX 1 lần duy nhất, merge vào df_raw
        # Tất cả lời gọi prepare_data_from_df() sau đó dùng lại cột này
        # — tránh 300+ lần gọi yfinance trong GA loop
        print("--- Đang tải dữ liệu Macro (VIX, TNX) ---")
        start_str = df_raw.index[0].strftime('%Y-%m-%d')
        end_str   = df_raw.index[-1].strftime('%Y-%m-%d')
        macro_df  = fetch_macro_data(TICKER_SYMBOL, start_str, end_str, df_raw.index)
        df_raw['VIX'] = macro_df['VIX']
        df_raw['TNX'] = macro_df['TNX']
        print(f"  VIX/TNX đã merge vào df_raw. NaN còn lại: {df_raw[['VIX','TNX']].isna().sum().sum()}")

        best_params, fitness_history, comp_history, generation_stats = run_ga_lstm(df_raw)

        # Vẽ convergence plot mới (có breakdown components)
        plot_convergence(fitness_history, comp_history, generation_stats)

        f_units, f_dropout, f_lr, f_batch, f_window, f_filters, f_layers = best_params

        print(f"\n{'='*55}")
        print("  Best chromosome tìm được:")
        print(f"    Units={f_units}, Dropout={f_dropout}, LR={f_lr}")
        print(f"    Batch={f_batch}, Window={f_window}, Filters={f_filters}, Layers={f_layers}")
        print(f"{'='*55}")

        print("\nĐang huấn luyện mô hình cuối cùng...")
        X_train, y_train, X_val, y_val, X_test, y_test, _, scaler_y = prepare_train_validation_test(
            df_raw, f_window, save_scalers=True   # lưu scaler_x/y.pkl
        )
        num_features = X_train.shape[2]   # (n, window, features) → dim 2

        final_model = CNN_LSTM(
            input_size        = num_features,
            hidden_layer_size = f_units,
            dropout_rate      = f_dropout,
            cnn_filters       = f_filters,
            num_layers        = f_layers,
        ).to(device)

        training_result = train_with_early_stopping(
            final_model, X_train, y_train, X_val, y_val,
            f_lr, f_batch, device, max_epochs=120, patience=15,
        )
        final_model = training_result.model
        print(
            f"  Restored epoch {training_result.best_epoch}; "
            f"stopped after {training_result.epochs_ran} epochs"
        )

        final_model.eval()
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = final_model(X_test_t).cpu().numpy()

        # Target của mô hình là daily return, nên metric được báo cáo trên return,
        # không phải USD.
        y_test_return = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        pred_return   = scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()

        rmse = np.sqrt(mean_squared_error(y_test_return, pred_return))
        mae  = mean_absolute_error(y_test_return, pred_return)
        directional_accuracy = compute_directional_accuracy(
            y_test_return,
            pred_return,
        )

        print(f"\n  RMSE (return): {rmse:.6f}")
        print(f"  MAE  (return): {mae:.6f}")
        print(f"  Directional Accuracy: {directional_accuracy * 100:.2f}%")

        # Lưu kết quả experiment thực tế để benchmark/reproduce.
        os.makedirs("experiments", exist_ok=True)
        result_row = pd.DataFrame([{
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "ticker": TICKER_SYMBOL,
            "model": "GA-WOA CNN-LSTM-Attention",
            "rmse_return": float(rmse),
            "mae_return": float(mae),
            "directional_accuracy": float(directional_accuracy),
            "units": int(f_units),
            "dropout": float(f_dropout),
            "learning_rate": float(f_lr),
            "batch_size": int(f_batch),
            "window_size": int(f_window),
            "cnn_filters": int(f_filters),
            "num_layers": int(f_layers),
            "best_epoch": training_result.best_epoch,
        }])
        results_path = "experiments/results.csv"
        result_row.to_csv(
            results_path,
            mode="a",
            header=not os.path.exists(results_path),
            index=False,
        )
        print(f"  Đã ghi kết quả vào {results_path}")

        # Lưu config + model
        model_config = {
            "input_size"       : int(num_features),
            "hidden_layer_size": int(f_units),
            "dropout_rate"     : float(f_dropout),
            "cnn_filters"      : int(f_filters),
            "num_layers"       : int(f_layers),
            "window_size"      : int(f_window),
            "learning_rate"    : float(f_lr),
            "batch_size"       : int(f_batch),
            # [MỚI] Lưu lại fitness weights đã dùng để reproduce
            "fitness_weights"  : FITNESS_WEIGHTS,
        }
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/model_config.json", "w") as f:
            json.dump(model_config, f, indent=2)

        torch.save(final_model.state_dict(), "artifacts/best_model.pth")
        print("\nĐã lưu artifacts/model_config.json và artifacts/best_model.pth. Web App sẵn sàng!")
