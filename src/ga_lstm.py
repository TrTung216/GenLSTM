import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import các hàm từ file khác
from data_prep import prepare_data
from fitness_function import evaluate_fitness, CNN_LSTM, device

TICKER_SYMBOL = 'AAPL'
START_DATE = '2015-01-01'
END_DATE = '2026-01-01'

# ==========================================
# 1. ĐỊNH NGHĨA KHÔNG GIAN TÌM KIẾM MỞ RỘNG (6 GEN)
# ==========================================
SPACE_UNITS = [32, 64, 96, 128]
SPACE_DROPOUT = [0.05, 0.1, 0.15]
SPACE_LR = [0.0001, 0.0005, 0.001, 0.005, 0.01]
SPACE_BATCH = [16, 32, 64]
SPACE_WINDOW = [10, 14, 20, 30] # GEN MỚI: Độ dài chuỗi thời gian nhìn về quá khứ
SPACE_FILTERS = [16, 32, 64] # GEN MỚI: Số lượng bộ lọc CNN

POPULATION_SIZE = 20
GENERATIONS = 15
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 3

# ==========================================
# 2. CÁC TOÁN TỬ DI TRUYỀN (ĐÃ CẬP NHẬT CHO 6 GEN)
# ==========================================

def create_individual():
    """Khởi tạo một cá thể với 6 Gene"""
    return [
        random.choice(SPACE_UNITS),
        random.choice(SPACE_DROPOUT),
        random.choice(SPACE_LR),
        random.choice(SPACE_BATCH),
        random.choice(SPACE_WINDOW),
        random.choice(SPACE_FILTERS)
    ]

def initial_population(pop_size):
    return [create_individual() for _ in range(pop_size)]

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
    """Lai ghép 1 điểm cắt cho NST 6 Gene"""
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, 5) # Cắt ngẫu nhiên từ gene 1 đến 5
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1, parent2

def mutate(individual):
    """Đột biến ngẫu nhiên 1 gene (Reset Mutation)"""
    if random.random() < MUTATION_RATE:
        gene_idx = random.randint(0, 5) 
        if gene_idx == 0:
            individual[0] = random.choice(SPACE_UNITS)
        elif gene_idx == 1:
            individual[1] = random.choice(SPACE_DROPOUT)
        elif gene_idx == 2:
            individual[2] = random.choice(SPACE_LR)
        elif gene_idx == 3:
            individual[3] = random.choice(SPACE_BATCH)
        elif gene_idx == 4:
            individual[4] = random.choice(SPACE_WINDOW)
        elif gene_idx == 5:
            individual[5] = random.choice(SPACE_FILTERS)
    return individual

# ==========================================
# 3. VÒNG LẶP TIẾN HÓA
# ==========================================
def run_ga_lstm():
    print("Bắt đầu quá trình tối ưu hóa GA-CNN-LSTM...")
    
    population = initial_population(POPULATION_SIZE)
    best_chromosome_overall = None
    best_fitness_overall = -1
    history_best_fitness = [] 
    
    for gen in range(GENERATIONS):
        print(f"\n--- Thế hệ {gen + 1}/{GENERATIONS} ---")
        
        fitness_scores = []
        for i, chromosome in enumerate(population):
            # Lấy Gene quy định Window Size
            window_size = chromosome[4]
            
            # GỌI DỮ LIỆU ĐỘNG: Mỗi cá thể có thể có bộ dữ liệu với Window Size khác nhau
            X_train, y_train, X_test, y_test, _ = prepare_data(TICKER_SYMBOL, START_DATE, END_DATE, window_size)
            
            val_size = int(len(X_train) * 0.2)
            X_train_ga, y_train_ga = X_train[:-val_size], y_train[:-val_size]
            X_val_ga, y_val_ga = X_train[-val_size:], y_train[-val_size:]
            
            # Truyền toàn bộ 6 Gene vào hàm evaluate_fitness
            fitness = evaluate_fitness(chromosome, X_train_ga, y_train_ga, X_val_ga, y_val_ga)
            fitness_scores.append(fitness)
            print(f"Cá thể {i+1} {chromosome} -> Fitness: {fitness:.4f}")
            
        gen_best_idx = np.argmax(fitness_scores)
        gen_best_chromosome = population[gen_best_idx]
        gen_best_fitness = fitness_scores[gen_best_idx]
        
        print(f"Tốt nhất thế hệ {gen + 1}: {gen_best_chromosome} (Fitness: {gen_best_fitness:.4f})")
        history_best_fitness.append(gen_best_fitness)
        
        if gen_best_fitness > best_fitness_overall:
            best_fitness_overall = gen_best_fitness
            best_chromosome_overall = copy.deepcopy(gen_best_chromosome)
            
        if gen == GENERATIONS - 1:
            break
            
        selected_population = tournament_selection(population, fitness_scores)
        
        next_generation = []
        for i in range(0, POPULATION_SIZE, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i+1] if i+1 < POPULATION_SIZE else selected_population[0]
            child1, child2 = crossover(parent1, parent2)
            next_generation.extend([child1, child2])
            
        next_generation = [mutate(ind) for ind in next_generation]
        next_generation[0] = copy.deepcopy(gen_best_chromosome)
        population = next_generation
        
    print("\n==========================================")
    print("KẾT QUẢ TỐI ƯU HÓA CUỐI CÙNG")
    print(f"Tham số tốt nhất: Units={best_chromosome_overall[0]}, Dropout={best_chromosome_overall[1]}, LR={best_chromosome_overall[2]}, Batch={best_chromosome_overall[3]}, Window={best_chromosome_overall[4]}, Filters={best_chromosome_overall[5]}")
    print(f"Điểm Fitness cao nhất: {best_fitness_overall:.4f}")
    
    return best_chromosome_overall, history_best_fitness

# ==========================================
# 4. CHẠY HỆ THỐNG VÀ TRỰC QUAN HÓA KẾT QUẢ
# ==========================================
if __name__ == "__main__":
    
    # Chạy GA để tìm tham số (và Window Size) tốt nhất
    best_params, fitness_history = run_ga_lstm()
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, GENERATIONS + 1), fitness_history, marker='o', linestyle='-', color='r')
    plt.title('Đường cong hội tụ của GA (6 Genes)')
    plt.xlabel('Thế hệ (Generation)')
    plt.ylabel('Max Fitness')
    plt.grid(True)
    plt.show()
    
    # --- ĐÁNH GIÁ MÔ HÌNH CUỐI CÙNG ---
    print("\nĐang huấn luyện Mô hình cuối cùng với bộ tham số tốt nhất...")
    final_units, final_dropout, final_lr, final_batch, final_window, final_filters = best_params
    
    # Lấy lại bộ dữ liệu với Window Size tốt nhất mà GA vừa tìm được
    X_train, y_train, X_test, y_test, scaler = prepare_data(TICKER_SYMBOL, START_DATE, END_DATE, final_window)
    
    # Tự động đếm số lượng tính năng thực tế từ dữ liệu
    num_features = X_train.shape[2] 
    
    # KHÔNG CÒN GÕ CỨNG SỐ KÊNH, TỰ ĐỘNG LẤY TỪ GA VÀ DỮ LIỆU
    final_model = CNN_LSTM(input_size=num_features, hidden_layer_size=final_units, dropout_rate=final_dropout, cnn_filters=final_filters).to(device)
    
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(final_model.parameters(), lr=final_lr)
    
    X_train_full_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_full_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    train_full_data = TensorDataset(X_train_full_t, y_train_full_t)
    train_full_loader = DataLoader(train_full_data, batch_size=final_batch, shuffle=False)
    
    final_model.train()
    for epoch in range(50):
        for seq, labels in train_full_loader:
            optimizer.zero_grad()
            y_pred = final_model(seq)
            loss = loss_function(y_pred, labels)
            loss.backward()
            optimizer.step()
            
    final_model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = final_model(X_test_t).cpu().numpy()
    
    predictions_usd = scaler.inverse_transform(predictions)
    y_test_usd = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    final_rmse = np.sqrt(mean_squared_error(y_test_usd, predictions_usd))
    print(f"\nSai số RMSE trên tập Test (USD): {final_rmse:.2f}")
    
    plt.figure(figsize=(14, 6))
    plt.plot(y_test_usd, color='blue', label='Giá thực tế')
    plt.plot(predictions_usd, color='red', linestyle='dashed', label='Giá dự đoán (GA 6 Genes)')
    plt.title(f'Dự báo giá cổ phiếu {TICKER_SYMBOL}')
    plt.xlabel('Thời gian')
    plt.ylabel('Giá (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()