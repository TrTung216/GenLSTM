import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Import các biến dữ liệu từ file data_prep.py
from data_prep import X_train, y_train, X_test, y_test, scaler, TICKER_SYMBOL

# Import hàm đánh giá fitness từ file fitness_function.py
from fitness_function import evaluate_fitness
# ==========================================
# 1. ĐỊNH NGHĨA KHÔNG GIAN TÌM KIẾM
# ==========================================
SPACE_UNITS = [16, 32, 64, 128]
SPACE_DROPOUT = [0.05, 0.1, 0.15, 0.2]
SPACE_LR = [0.001, 0.005, 0.01]
SPACE_BATCH = [16, 32, 64]

# Thông số GA
POPULATION_SIZE = 20
GENERATIONS = 15
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 3 # Chọn ngẫu nhiên 3 cá thể để thi đấu, lấy con khỏe nhất

# ==========================================
# 2. CÁC TOÁN TỬ DI TRUYỀN (GENETIC OPERATORS)
# ==========================================

def create_individual():
    """Khởi tạo một cá thể (Nhiễm sắc thể) ngẫu nhiên từ không gian tìm kiếm"""
    return [
        random.choice(SPACE_UNITS),
        random.choice(SPACE_DROPOUT),
        random.choice(SPACE_LR),
        random.choice(SPACE_BATCH)
    ]

def initial_population(pop_size):
    """Khởi tạo quần thể ban đầu"""
    return [create_individual() for _ in range(pop_size)]

def tournament_selection(population, fitness_scores):
    """Chọn lọc cá thể xuất sắc bằng phương pháp Tournament"""
    selected_population = []
    for _ in range(len(population)):
        # Bốc ngẫu nhiên 3 cá thể (chỉ lấy chỉ số index của chúng)
        tournament_indices = random.sample(range(len(population)), TOURNAMENT_SIZE)
        
        # Tìm cá thể có điểm fitness cao nhất trong 3 con
        best_idx = tournament_indices[0]
        for idx in tournament_indices[1:]:
            if fitness_scores[idx] > fitness_scores[best_idx]:
                best_idx = idx
                
        selected_population.append(copy.deepcopy(population[best_idx]))
    return selected_population

def crossover(parent1, parent2):
    """Lai ghép 1 điểm cắt (Single-point Crossover)"""
    if random.random() < CROSSOVER_RATE:
        # Chọn ngẫu nhiên 1 điểm cắt từ gene thứ 1 đến gene thứ 3
        point = random.randint(1, 3) 
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1, parent2

def mutate(individual):
    """Đột biến ngẫu nhiên 1 gene (Reset Mutation)"""
    if random.random() < MUTATION_RATE:
        # Chọn ngẫu nhiên vị trí gene bị đột biến
        gene_idx = random.randint(0, 3) 
        
        if gene_idx == 0:
            individual[0] = random.choice(SPACE_UNITS)
        elif gene_idx == 1:
            individual[1] = random.choice(SPACE_DROPOUT)
        elif gene_idx == 2:
            individual[2] = random.choice(SPACE_LR)
        elif gene_idx == 3:
            individual[3] = random.choice(SPACE_BATCH)
            
    return individual

# ==========================================
# 3. VÒNG LẶP TIẾN HÓA (MAIN GA LOOP)
# ==========================================
def run_ga_lstm(X_train, y_train, X_val, y_val):
    print("Bắt đầu quá trình tối ưu hóa GA-LSTM...")
    
    population = initial_population(POPULATION_SIZE)
    best_chromosome_overall = None
    best_fitness_overall = -1
    
    # Lưu lại lịch sử để vẽ biểu đồ cho báo cáo NCKH
    history_best_fitness = [] 
    
    for gen in range(GENERATIONS):
        print(f"\n--- Thế hệ {gen + 1}/{GENERATIONS} ---")
        
        # 3.1. Đánh giá độ thích nghi của toàn bộ quần thể
        fitness_scores = []
        for i, chromosome in enumerate(population):
            # Hàm evaluate_fitness là hàm chúng ta đã viết ở bước trước
            fitness = evaluate_fitness(chromosome, X_train, y_train, X_val, y_val)
            fitness_scores.append(fitness)
            print(f"Cá thể {i+1} {chromosome} -> Fitness: {fitness:.4f}")
            
        # 3.2. Tìm cá thể vô địch của thế hệ này (Chủ nghĩa tinh hoa - Elitism)
        gen_best_idx = np.argmax(fitness_scores)
        gen_best_chromosome = population[gen_best_idx]
        gen_best_fitness = fitness_scores[gen_best_idx]
        
        print(f"Tốt nhất thế hệ {gen + 1}: {gen_best_chromosome} (Fitness: {gen_best_fitness:.4f})")
        history_best_fitness.append(gen_best_fitness)
        
        # Cập nhật nhà vô địch toàn cục
        if gen_best_fitness > best_fitness_overall:
            best_fitness_overall = gen_best_fitness
            best_chromosome_overall = copy.deepcopy(gen_best_chromosome)
            
        # Nếu là thế hệ cuối cùng thì dừng, không sinh sản nữa
        if gen == GENERATIONS - 1:
            break
            
        # 3.3. Chọn lọc (Selection)
        selected_population = tournament_selection(population, fitness_scores)
        
        # 3.4. Lai ghép (Crossover)
        next_generation = []
        for i in range(0, POPULATION_SIZE, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i+1] if i+1 < POPULATION_SIZE else selected_population[0]
            
            child1, child2 = crossover(parent1, parent2)
            next_generation.append(child1)
            next_generation.append(child2)
            
        # 3.5. Đột biến (Mutation)
        next_generation = [mutate(ind) for ind in next_generation]
        
        # 3.6. Chủ nghĩa Tinh hoa (Elitism): Giữ lại con mạnh nhất của thế hệ trước thay cho 1 con ngẫu nhiên
        # Giúp đảm bảo đồ thị tiến hóa không bao giờ bị tụt lùi
        next_generation[0] = copy.deepcopy(gen_best_chromosome)
        
        # Đưa thế hệ mới vào vòng lặp tiếp theo
        population = next_generation
        
    print("\n==========================================")
    print("KẾT QUẢ TỐI ƯU HÓA CUỐI CÙNG")
    print(f"Tham số tốt nhất: Units={best_chromosome_overall[0]}, Dropout={best_chromosome_overall[1]}, LR={best_chromosome_overall[2]}, Batch={best_chromosome_overall[3]}")
    print(f"Điểm Fitness cao nhất: {best_fitness_overall:.4f}")
    
    return best_chromosome_overall, history_best_fitness
# ==========================================
# 4. CHẠY HỆ THỐNG VÀ TRỰC QUAN HÓA KẾT QUẢ (PYTORCH)
# ==========================================
if __name__ == "__main__":
    # Import class PyTorchLSTM và device từ fitness_function để dùng cho mô hình cuối
    from fitness_function import PyTorchLSTM, device
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    # 4.1. Tách tập Validation
    val_size = int(len(X_train) * 0.2)
    X_train_ga, y_train_ga = X_train[:-val_size], y_train[:-val_size]
    X_val_ga, y_val_ga = X_train[-val_size:], y_train[-val_size:]
    
    # 4.2. Chạy GA
    best_params, fitness_history = run_ga_lstm(X_train_ga, y_train_ga, X_val_ga, y_val_ga)
    
    # 4.3. Vẽ biểu đồ hội tụ
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, GENERATIONS + 1), fitness_history, marker='o', linestyle='-', color='r')
    plt.title('Đường cong hội tụ của Thuật toán Di truyền (GA)')
    plt.xlabel('Thế hệ (Generation)')
    plt.ylabel('Độ thích nghi cao nhất (Max Fitness)')
    plt.grid(True)
    plt.show()
    
    # 4.4. Xây dựng MÔ HÌNH CUỐI CÙNG bằng PyTorch
    print("\n🚀 Đang huấn luyện Mô hình cuối cùng với bộ tham số tốt nhất...")
    final_units, final_dropout, final_lr, final_batch = best_params
    
    final_model = PyTorchLSTM(input_size=1, hidden_layer_size=final_units, dropout_rate=final_dropout).to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(final_model.parameters(), lr=final_lr)
    
    # Chuẩn bị dữ liệu Train (Gộp toàn bộ)
    X_train_full_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_full_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    train_full_data = TensorDataset(X_train_full_t, y_train_full_t)
    train_full_loader = DataLoader(train_full_data, batch_size=final_batch, shuffle=False)
    
    # Huấn luyện mô hình cuối cùng (50 epochs)
    final_model.train()
    for epoch in range(50):
        for seq, labels in train_full_loader:
            optimizer.zero_grad()
            y_pred = final_model(seq)
            loss = loss_function(y_pred, labels)
            loss.backward()
            optimizer.step()
            
    # 4.5. Dự đoán trên tập TEST
    final_model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = final_model(X_test_t).cpu().numpy()
    
    # Đảo ngược chuẩn hóa
    predictions_usd = scaler.inverse_transform(predictions)
    y_test_usd = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    final_rmse = np.sqrt(mean_squared_error(y_test_usd, predictions_usd))
    print(f"\nSai số RMSE trên tập Test (USD): {final_rmse:.2f}")
    
    # 4.6. Vẽ biểu đồ So sánh
    plt.figure(figsize=(14, 6))
    plt.plot(y_test_usd, color='blue', label='Giá thực tế (Actual Price)')
    plt.plot(predictions_usd, color='red', linestyle='dashed', label='Giá dự đoán (Predicted Price)')
    plt.title(f'Dự báo giá cổ phiếu {TICKER_SYMBOL} bằng mô hình GA-LSTM (PyTorch)')
    plt.xlabel('Thời gian (Ngày)')
    plt.ylabel('Giá (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()