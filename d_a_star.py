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
        population_size = 4
        generations =1
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
        return(greatest_of_all_time)

        print('new_generation',new_generation)
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
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0

        while frontier:
            current = heapq.heappop(frontier)[1]

            if current == goal:
                break

            for next in self.neighbors(current):
                new_cost = cost_so_far[current] + self.cost(current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next)
                    heapq.heappush(frontier, (priority, next))
                    came_from[next] = current

        if goal not in came_from:
            return []  # Return an empty list if there is no path to the goal

        # Reconstruct the path
        current = goal
        path = []
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()

        return path
    

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
        distance_matrix = [[0 for _ in range(len(list_start))] for _ in range(len(list_start))]
        path_matrix = [[[] for _ in range(len(list_start))] for _ in range(len(list_start))]

        for i in range(len(list_start)):
            for j in range(i + 1, len(list_start)):
                path = self.genetic_algorithm(list_start[i], list_start[j])
                distance = 0
                for k in range(1, len(path)):
                    dx = abs(path[k][0] - path[k-1][0])
                    dy = abs(path[k][1] - path[k-1][1])
                    if dx == 1 and dy == 1:
                        distance += 2**0.5  # Diagonal movement
                    else:
                        distance += 1  # Horizontal or vertical movement

                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance
                path_matrix[i][j] = path
                path_matrix[j][i] = path[::-1]

        return distance_matrix, path_matrix, list_start