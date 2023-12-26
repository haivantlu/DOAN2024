import random


file_path = 'Test/TestCase1/map_20_5s_2.txt'  # Đặt tên file và đường dẫn tương ứng

# Mở file để đọc
with open(file_path, 'r') as file:
    # Đọc kích thước ma trận từ dòng đầu tiên
    size = int(file.readline().strip())
    
    # Khai báo mảng hai chiều (ma trận) và khởi tạo nó với giá trị mặc định là 0
    grid = [[0 for _ in range(size)] for _ in range(size)]
    
    # Đọc từng dòng trong file và cập nhật giá trị của ma trận
    for i in range(size):
        line = file.readline().strip().split()
        grid[i] = [int(cell) for cell in line]


# Cài đặt thuật toán GA
def genetic_algorithm(start, goal):
    population_size = 5000
    generations = 2000
    mutation_rate = 0.1

    # Tạo quần thể ban đầu
    population = [create_individual(start,goal) for _ in range(population_size)]

    for _ in range(generations):
        # Đánh giá và chọn lọc cá thể tốt nhất
        fitness_scores = [fitness(individual) for individual in population]
        return population[fitness_scores.index(min(fitness_scores))]  # Trả về ngay khi tìm thấy lời giải

        min_value = min(fitness_scores)  # Tìm giá trị nhỏ nhất trong list
        count = fitness_scores.count(min_value)  # Đếm số lượng giá trị nhỏ nhất
        if count == 1:
            return population[fitness_scores.index(min(fitness_scores))]  # Trả về ngay khi tìm thấy lời giải

        selected_individuals = selection(population, fitness_scores)

        # Tạo thế hệ mới từ cá thể được chọn lọc
        new_generation = []
        while len(new_generation) < population_size:
            parent1, parent2 = random.choice(selected_individuals), random.choice(selected_individuals)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_generation.extend([child1, child2])

        # Thay thế quần thể cũ bằng thế hệ mới
        population = new_generation

    return None  # Trả về None nếu không tìm thấy lời giải

# Cài đặt hàm tạo cá thể ngẫu nhiên
def create_individual(start,goal):
    x, y = start
    individual = [(x, y)]
    visited_coordinates = set({(x, y)})
    # print( visited_coordinates)

    while (x, y) != goal:
        possible_moves = []

        if x > 0 and (x-1, y) not in visited_coordinates and grid[x-1][y] != 1:
            possible_moves.append((x-1, y))
        if x < len(grid) - 1 and (x+1, y) not in visited_coordinates and grid[x+1][y] != 1:
            possible_moves.append((x+1, y))
        if y > 0 and (x, y-1) not in visited_coordinates and grid[x][y-1] != 1:
            possible_moves.append((x, y-1))
        if y < len(grid[0]) - 1 and (x, y+1) not in visited_coordinates and grid[x][y+1] != 1:
            possible_moves.append((x, y+1))

        if not possible_moves:
            return []  # Tránh lặp vô hạn nếu không có bước đi hợp lệ

        next_move = random.choice(possible_moves)
        x, y = next_move
        individual.append(next_move)
        visited_coordinates.add((x, y))

    # print("Cá thể:", individual)
    return individual

# Kiểm tra xem hướng đi có hợp lệ không
def is_valid_move(direction, x, y):
    if direction == 'Lên' and x > 1 and grid[x-1][y] != 1:
        return True
    elif direction == 'Xuống' and x < len(grid) - 1 and grid[x+1][y] != 1:
        return True
    elif direction == 'Trái' and y > 1 and grid[x][y-1] != 1:
        return True
    elif direction == 'Phải' and y < len(grid[0]) - 1 and grid[x][y+1] != 1:
        return True
    return False

# Đánh giá độ thích nghi của cá thể
# def fitness(individual):
#     x, y = start
#     visited_coordinates = set()

#     for move in individual:
#         if move not in visited_coordinates:
#             visited_coordinates.add(move)
#             x, y = move

#             if move == goal:
#                 return 1.0  # Đạt được đích

#     return 1.0 / (abs(goal[0] - x) + abs(goal[1] - y) + 1)  # Khoảng cách còn lại

# Cài đặt hàm đánh giá sức mạnh (fitness) theo độ dài đường đi

def fitness(individual):
    if len(individual) == 0:
        return 9999
    return len(individual)

# Chọn lọc cá thể dựa trên đánh giá sức mạnh
def selection(population, fitness_scores):
    min_fitness  = min(fitness_scores)
    selected_individuals = [individual for individual, fitness_score in zip(population, fitness_scores) if fitness_score == min_fitness]
    # print("selected_individuals:", selected_individuals)
    return selected_individuals
    
  

# Lai ghép giữa hai cá thể
def crossover(parent1, parent2):
    split_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child1 = parent1[:split_point] + parent2[split_point:]
    child2 = parent2[:split_point] + parent1[split_point:]
    return child1, child2

# Đột biến cá thể
def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            new_move = (random.randint(0, len(grid)-1), random.randint(0, len(grid[0])-1))
            # Kiểm tra xem bước đi mới có hợp lệ không
            if new_move not in individual and grid[new_move[0]][new_move[1]] != 1:
                individual[i] = new_move
    return individual



# start = (0, 10)
# goal = 19, 16
# # Thực thi
# solution = genetic_algorithm(start, goal)
# print("Solution:", solution)

list_start = []
for i in range(len(grid)):
    for j in range(len(grid[0])):
        if grid[i][j] == 2:
            list_start.append((i, j))

print(list_start)

# Khai báo distance_matrix là một mảng hai chiều có kích thước len(list_start) x len(list_start)
distance_matrix = [[0 for _ in range(len(list_start))] for _ in range(len(list_start))]

for i in range(len(list_start)):
    for j in range(i + 1, len(list_start)):  # Chỉ tính cho một nửa ma trận
        distance = len(genetic_algorithm(list_start[i], list_start[j]))
        distance_matrix[i][j] = distance
        distance_matrix[j][i] = distance  # Đảm bảo ma trận đối xứng qua đường chéo

print(distance_matrix)

