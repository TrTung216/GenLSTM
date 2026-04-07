import numpy as np
import random
import copy
import math
import matplotlib.pyplot as plt
import json
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import pandas_market_calendars as mcal
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta
import yfinance as yf

# Import các hàm từ file khác
from data_prep import prepare_data_from_df, download_stock_data
from fitness_function import evaluate_fitness, CNN_LSTM, device

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Current Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

TICKER_SYMBOL = 'AAPL'
START_DATE = '2015-01-01'

# 1. Tự động lấy ngày giao dịch cuối cùng từ NYSE
nyse = mcal.get_calendar('NYSE')
now = datetime.now()
schedule = nyse.schedule(start_date=(now - timedelta(days=10)).strftime('%Y-%m-%d'), 
                         end_date=now.strftime('%Y-%m-%d'))
END_DATE = schedule.index[-1].strftime('%Y-%m-%d')
print(f"Ngày giao dịch cuối cùng xác định bởi NYSE: {END_DATE}")

# ==========================================
# 2. KHÔNG GIAN TÌM KIẾM MỞ RỘNG (7 GENES)
# ==========================================
SPACE_UNITS = [32, 64, 96, 128]
SPACE_DROPOUT = [0.05, 0.1, 0.15]
SPACE_LR = [0.0001, 0.0005, 0.001, 0.005]
SPACE_BATCH = [16, 32, 64]
SPACE_WINDOW = [20, 30, 45, 60] 
SPACE_FILTERS = [16, 32, 64]
SPACE_LAYERS = [1, 2] 

SPACES_DICT = {
    'SPACE_UNITS': SPACE_UNITS,
    'SPACE_DROPOUT': SPACE_DROPOUT,
    'SPACE_LR': SPACE_LR,
    'SPACE_BATCH': SPACE_BATCH,
    'SPACE_WINDOW': SPACE_WINDOW,
    'SPACE_FILTERS': SPACE_FILTERS,
    'SPACE_LAYERS': SPACE_LAYERS
}

POPULATION_SIZE = 20
GENERATIONS = 15
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 3 # Giữ nguyên hoặc có thể giảm xuống 2 nếu vẫn thấy kẹt

def create_individual():
    """Khởi tạo một cá thể với 7 Gene"""
    return [
        random.choice(SPACE_UNITS),
        random.choice(SPACE_DROPOUT),
        random.choice(SPACE_LR),
        random.choice(SPACE_BATCH),
        random.choice(SPACE_WINDOW),
        random.choice(SPACE_FILTERS),
        random.choice(SPACE_LAYERS)
    ]

def tournament_selection(population, fitness_scores):
    selected_population = []
    for _ in range(len(population)):
        tournament_indices = random.sample(range(len(population)), TOURNAMENT_SIZE)
        best_idx = tournament_indices[0]
        for idx in tournament_indices[1:]:
            if fitness_scores[idx] > fitness_scores[best_idx]:
                best_idx = idx
        selected_population.append(copy.deepcopy(population[best_idx]))
    return selected_population

def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, 6) 
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    return parent1, parent2

def mutate(individual, mut_rate):
    """Đột biến với tỷ lệ động (Dynamic Mutation)"""
    if random.random() < mut_rate:
        gene_idx = random.randint(0, 6)
        choices = [SPACE_UNITS, SPACE_DROPOUT, SPACE_LR, SPACE_BATCH, SPACE_WINDOW, SPACE_FILTERS, SPACE_LAYERS]
        individual[gene_idx] = random.choice(choices[gene_idx])
    return individual

def woa_refinement(chromosome, best_chromosome, current_gen, max_gen, spaces):
    """Toán tử Tinh chỉnh cục bộ của Thuật toán Đàn Cá Voi (WOA)"""
    new_chrom = list(chromosome)
    
    # Tham số a giảm tuyến tính từ 2 về 0 để siết chặt vòng vây
    a = 2 - current_gen * (2 / max_gen)
    
    for i in range(len(chromosome)):
        p = random.random()
        
        # WOA Math Operations
        if p < 0.5:
            # Thu hẹp vòng vây (Shrinking Encircling)
            r = random.random()
            A = 2 * a * r - a
            C = 2 * r
            D = abs(C * best_chromosome[i] - chromosome[i])
            new_val = best_chromosome[i] - A * D
        else:
            # Lưới bong bóng xoắn ốc (Spiral Bubble-net)
            D_prime = abs(best_chromosome[i] - chromosome[i])
            b = 1.0 
            l = random.uniform(-1, 1)
            new_val = D_prime * math.exp(b * l) * math.cos(2 * math.pi * l) + best_chromosome[i]

        # Snap (Gióng) giá trị toán học về lại không gian tham số hợp lệ
        if i == 1: # Dropout
            new_chrom[i] = max(0.0, min(0.5, round(new_val, 2)))
        elif i == 2: # Learning Rate
            new_chrom[i] = max(0.0001, min(0.01, new_val))
            new_chrom[i] = min(spaces['SPACE_LR'], key=lambda x: abs(x - new_chrom[i]))
        else: # Biến rời rạc
            mapping = {
                0: spaces['SPACE_UNITS'],
                3: spaces['SPACE_BATCH'],
                4: spaces['SPACE_WINDOW'],
                5: spaces['SPACE_FILTERS'],
                6: spaces['SPACE_LAYERS']
            }
            if i in mapping:
                valid_options = mapping[i]
                new_chrom[i] = min(valid_options, key=lambda x: abs(x - new_val))

    return new_chrom

# ==========================================
# 3. VÒNG LẶP TIẾN HÓA (GA-WOA)
# ==========================================
def run_ga_lstm(df_raw):
    print(f" Bắt đầu lai ghép GA-WOA với {len(df_raw)} dòng dữ liệu.")
    
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    best_chromosome_overall = None
    best_fitness_overall = -1
    history_best_fitness = [] 
    
    for gen in range(GENERATIONS):
        print(f"\n--- Thế hệ {gen + 1}/{GENERATIONS} ---")
        fitness_scores = []
        
        for i, chromosome in enumerate(population):
            data_package = prepare_data_from_df(df_raw, chromosome[4])
            
            if data_package[0] is None or len(data_package[0]) == 0:
                fitness = 1e-6 
            else:
                X_train, y_train, X_test, y_test, _ = data_package
                val_size = int(len(X_train) * 0.2)
                if val_size <= 0: 
                    fitness = 1e-6
                else:
                    X_train_ga, y_train_ga = X_train[:-val_size], y_train[:-val_size]
                    X_val_ga, y_val_ga = X_train[-val_size:], y_train[-val_size:]
                    fitness = evaluate_fitness(chromosome, X_train_ga, y_train_ga, X_val_ga, y_val_ga)
            
            fitness_scores.append(fitness)
            print(f"Cá thể {i+1} {chromosome} -> Fitness: {fitness:.4f}")
            
        # Sắp xếp quần thể theo Fitness giảm dần (Giỏi nhất nằm ở index 0)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[idx] for idx in sorted_indices]
        fitness_scores = [fitness_scores[idx] for idx in sorted_indices]
        
        gen_best_idx = 0 # Vì đã sort nên index 0 luôn là tốt nhất của Gen này
        if fitness_scores[gen_best_idx] > best_fitness_overall:
            best_fitness_overall = fitness_scores[gen_best_idx]
            best_chromosome_overall = copy.deepcopy(population[gen_best_idx])
            
        history_best_fitness.append(best_fitness_overall)
        print(f" Tốt nhất hiện tại: {best_chromosome_overall} (Fitness: {best_fitness_overall:.4f})")
        
        # --- GA-WOA HYBRID STRATEGY ---
        next_gen = [copy.deepcopy(best_chromosome_overall)] # 1. Elitism: Giữ lại King
        
        # 2. WOA (Local Search): 30% cá thể giỏi nhất sẽ bao vây tinh chỉnh quanh King
        num_woa = int(0.3 * POPULATION_SIZE)
        for i in range(1, num_woa):
            refined_chrom = woa_refinement(population[i], best_chromosome_overall, gen, GENERATIONS, SPACES_DICT)
            next_gen.append(refined_chrom)
            
        # 3. GA (Global Search): 70% còn lại dùng Crossover & Dynamic Mutation
        selected = tournament_selection(population, fitness_scores)
        num_ga = POPULATION_SIZE - num_woa
        
        for i in range(0, num_ga, 2):
            p1 = selected[random.randint(0, len(selected)-1)]
            p2 = selected[random.randint(0, len(selected)-1)]
            c1, c2 = crossover(p1, p2)
            
            # Tỷ lệ đột biến động (Dynamic Mutation) dao động [0.1 - 0.5]
            dynamic_mut_rate = random.uniform(0.1, 0.5)
            
            next_gen.append(mutate(c1, dynamic_mut_rate))
            if len(next_gen) < POPULATION_SIZE:
                next_gen.append(mutate(c2, dynamic_mut_rate))
                
        population = next_gen[:POPULATION_SIZE] # Đảm bảo đúng số lượng quần thể
        
    return best_chromosome_overall, history_best_fitness

# ==========================================
# 4. HUẤN LUYỆN CUỐI CÙNG & XUẤT FILE
# ==========================================
if __name__ == "__main__":
    print("--- Đang tải dữ liệu gốc từ Yahoo Finance ---")
    df_raw = yf.download(TICKER_SYMBOL, start=START_DATE, end=END_DATE, progress=False)
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    if df_raw.empty or len(df_raw) < 100:
        print(" Lỗi: Không có dữ liệu hoặc dữ liệu quá ngắn. Kiểm tra Ticker!")
    else:
        best_params, fitness_history = run_ga_lstm(df_raw)
        
        f_units, f_dropout, f_lr, f_batch, f_window, f_filters, f_layers = best_params
        
        print("\nĐang huấn luyện Mô hình cuối cùng với Scheduler...")
        X_train, y_train, X_test, y_test, scaler_y = prepare_data_from_df(df_raw, f_window)
        num_features = X_train.shape[2] 
        
        final_model = CNN_LSTM(
            input_size=num_features, 
            hidden_layer_size=f_units, 
            dropout_rate=f_dropout, 
            cnn_filters=f_filters,
            num_layers=f_layers
        ).to(device)
        
        loss_function = nn.HuberLoss(delta=1.0)
        optimizer = optim.Adam(final_model.parameters(), lr=f_lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5)

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=f_batch, shuffle=False, pin_memory=True)
        
        for epoch in range(120):
            final_model.train()
            total_loss = 0
            for seq, labels in train_loader:
                seq, labels = seq.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = loss_function(final_model(seq), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            scheduler.step(avg_loss)
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/120, Loss: {avg_loss:.6f}")

        final_model.eval()
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = final_model(X_test_t).cpu().numpy()
            rmse = np.sqrt(mean_squared_error(scaler_y.inverse_transform(y_test.reshape(-1,1)), 
                                            scaler_y.inverse_transform(preds)))
        
        print(f"\n RMSE cuối cùng: {rmse:.2f} USD")

        model_config = {
            "input_size": int(num_features),
            "hidden_layer_size": int(f_units),
            "dropout_rate": float(f_dropout),
            "cnn_filters": int(f_filters),
            "num_layers": int(f_layers),
            "window_size": int(f_window)
        }
        with open('model_config.json', 'w') as f:
            json.dump(model_config, f)
        
        torch.save(final_model.state_dict(), 'best_model.pth')
        print(" Đã lưu model_config.json và best_model.pth. Web App đã sẵn sàng!")