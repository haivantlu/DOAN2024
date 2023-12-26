import random
import numpy as np
import math
import heapq


class GeneticAlgorithm:
    def __init__(self, grid):
        # self.grid = self.read_grid(file_path)
        self.grid =grid
        self.list_start = self.find_start_positions()

    def manhattan_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  



    def euclidean_distance(self,point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def find_start_positions(self):
        list_start = []
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if self.grid[i][j] == 2:
                    list_start.append((i, j))
        return list_start

    def genetic_algorithm(self, start, goal):
        population_size = 100
        generations =100
        # selection_size = round(population_size * 0.82)
        selection_size = 82
        mutation_rate = 0.1
        
        # Tạo quần thể ban đầu
        population = [ind for ind in [self.create_individual(start, goal) for _ in range(population_size)]]

        while generations > 0:
            # print('population',population)
            # Chọn lọc
            selected_individuals = self.selection(population,selection_size)           
            new_generation = []
            while len(new_generation) < population_size:
                if not selected_individuals:  # kiểm tra xem danh sách có trống không
                    break  # hoặc xử lý tình huống phù hợp
                parent1, parent2 = random.choice(selected_individuals), random.choice(selected_individuals)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, mutation_rate)
                child2 = self.mutate(child2, mutation_rate)
                new_generation.append(child1)
                new_generation.append(child2)
          
            if new_generation:  # kiểm tra xem danh sách có trống không
                greatest_of_all_time = new_generation[0]
                for element in new_generation:
                    if len(element) < len(greatest_of_all_time):
                        greatest_of_all_time = element
            else:
                # Xử lý trường hợp danh sách trống
                greatest_of_all_time = None  # hoặc giá trị phù hợp khác    

            population = new_generation
            generations -= 1
            # print('lần:', generations)
        if greatest_of_all_time is not None:
            for i in range(len(greatest_of_all_time)):
                # Convert the tuple to a list
                current_position = list(greatest_of_all_time[i])
                
                # Modify the list elements
                # current_position[0] += 0.5
                # current_position[1] += 0.5
                
                # Convert the list back to a tuple
                greatest_of_all_time[i] = tuple(current_position)
        else:
            greatest_of_all_time = []
        print('greatest_of_all_time',greatest_of_all_time)
        return(greatest_of_all_time)

        # print('new_generation',new_generation)
        # Thay thế quần thể cũ bằng thế hệ này  
       
        return None  # Trả về None nếu không tìm thấy lời giải


    def neighbors(self, current):
        x, y = current
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Right, down, left, up, and the diagonal directions
        neighbors = []

        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < len(self.grid) and 0 <= new_y < len(self.grid[0]):
                if self.grid[new_x][new_y] != 1:  # Check the target cell
                    if dx != 0 and dy != 0:  # If moving diagonally
                        if self.grid[x + dx][y] != 1 and self.grid[x][y + dy] != 1:  # Check the two adjacent cells
                            neighbors.append((new_x, new_y))
                    else:  # If moving horizontally or vertically
                        neighbors.append((new_x, new_y))

        return neighbors
    
    def cost(self, current, next):
        x1, y1 = current
        x2, y2 = next
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5  # Euclidean distance

    def heuristic(self, goal, next):
        x1, y1 = goal
        x2, y2 = next
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5  # Euclidean distance



    def create_individual(self, start, goal):
        x, y = start
        individual = [(x, y)]
        visited_coordinates = set([(x, y)])

        while (x, y) != goal:
            current_distance = self.euclidean_distance([x, y], [goal[0], goal[1]])
            possible_moves = []
            fallback_moves = []

            for i in range(-1, 2):
                for j in range(-1, 2):
                    new_x, new_y = x + i, y + j

                    if 0 <= new_x < len(self.grid) \
                            and 0 <= new_y < len(self.grid[0]) \
                            and self.grid[new_x][new_y] != 1 \
                            and (new_x, new_y) not in visited_coordinates:
                        # Check for obstacles when moving diagonally
                        if i != 0 and j != 0:
                            if self.grid[x + i][y] == 1 or self.grid[x][y + j] == 1:
                                continue
                        new_distance = self.euclidean_distance([new_x, new_y], [goal[0], goal[1]])
                        if new_distance < current_distance:
                            possible_moves.append((new_x, new_y))
                        else:
                            fallback_moves.append((new_x, new_y))

            if not possible_moves and not fallback_moves:
                return []
            if not possible_moves:
                next_move = random.choice(fallback_moves)
            else:
                next_move = random.choice(possible_moves)
            x, y = next_move
            individual.append(next_move)
            visited_coordinates.add((x, y))

        return individual

    def fitness(self, individual):
        if len(individual) == 0:
            return 9999
        return len(individual)

    def selection(self, population, selection_size):
        # lấy ra selection_size các phẩn tử có độ dài lớn hơn 0
        non_empty_arrays = [arr for arr in population if len(arr) > 0]

        selected_individuals = sorted(non_empty_arrays, key=lambda arr: self.fitness(arr))[:selection_size]
        return selected_individuals
        

    def crossover(self, parent1, parent2):
         
        # Tìm danh sách các điểm chung
        crossover_points = []
        for i in range(1, len(parent1)-1):
            for j in range (1, len(parent2)-1):
              if parent1[i] == parent2[j]:
                crossover_points.append((i,j))
        
        # Lai ghép nếu không có điểm chung
        if len(crossover_points) == 0:
            # if len(parent1) > len(parent2):
            #     return parent2
            # else: 
            #     return parent1
            return parent1,parent2

        # Lai ghép nếu có 1 điểm chung
        if len(crossover_points) == 1:
            child1 = parent1[:crossover_points[0][0]] + parent2[crossover_points[0][1]:]
            child2 = parent2[:crossover_points[0][1]] + parent1[crossover_points[0][0]:]
            return [child1, child2]

        # Lai ghép nếu có 2 điểm chung thì điểm đầu và điểm cuối
        else:
            child1 = parent1[:crossover_points[0][0]] + parent2[crossover_points[0][1]:crossover_points[-1][1]] + parent1[crossover_points[-1][0]:]
            child2 = parent2[:crossover_points[0][1]] + parent1[crossover_points[0][0]:crossover_points[-1][0]] + parent2[crossover_points[-1][1]:]

            return [child1, child2]

    def mutate(self, individual, mutation_rate=0.1):
        if random.random() < mutation_rate:
            # Bước 1: Chọn ngẫu nhiên một nút X từ cá thể đột biến làm gen đột biến.
            mutation_point = random.choice(range(1, len(individual) - 1))
            X = individual[mutation_point]

            # Bước 2: Xác định tập N gồm tất cả các nút tự do gần X
            N = [(i, j) for i in range(len(self.grid)) for j in range(len(self.grid[0])) 
                 if self.grid[i][j] != 1 and (i, j) not in individual 
                 and (self.manhattan_distance((i, j), individual[mutation_point - 1]) == 1 
                      or self.manhattan_distance((i, j), individual[mutation_point + 1]) == 1)]

            while N:
                # Bước 3: Chọn ngẫu nhiên một nút Y từ tập hợp N.
                Y = random.choice(N)

                # Bước 4: Nếu nút trước (và sau) X được kết nối với Y, thì Y được áp dụng để thay thế X
                if self.manhattan_distance(Y, individual[mutation_point - 1]) == 1 and self.manhattan_distance(Y, individual[mutation_point + 1]) == 1:
                    individual[mutation_point] = Y
                    return individual

                # Nếu không thì lặp lại Bước 3 và Bước 4 cho đến khi tìm thấy nút mong muốn hoặc quá trình tìm kiếm của tập N kết thúc.
                N.remove(Y)

        return individual
    
    def CalcMatrix(self):
        list_start = self.find_start_positions()
        # Khai báo distance_matrix là một mảng hai chiều có kích thước len(list_start) x len(list_start)
        distance_matrix = [[0 for _ in range(len(list_start))] for _ in range(len(list_start))]
        # Khai báo path_matrix để lưu trữ đường đi ngắn nhất giữa các điểm
        path_matrix = [[[] for _ in range(len(list_start))] for _ in range(len(list_start))]

        for i in range(len(list_start)):
            for j in range(i + 1, len(list_start)):  # Chỉ tính cho một nửa ma trận
                path = self.genetic_algorithm(list_start[i], list_start[j])
                distance = len(path)

                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance  # Đảm bảo ma trận đối xứng qua đường chéo
                path_matrix[i][j] = path
                path_matrix[j][i] = path[::-1]  # Đảm bảo ma trận đối xứng qua đường chéo

        return distance_matrix, path_matrix
