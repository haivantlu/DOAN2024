import random
import numpy as np
import math
import heapq
from grid_map import GM


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
        generations = 100
        selection_size = 30
        mutation_rate = 0.1
        max_unchanged_generations = 5
        GridMap = GM(self.grid)
        
        # Create initial population
        population = [self.create_individual(start, goal) for _ in range(population_size)]
        population.append(GridMap.create_individual(start, goal))

        best_solution = None
        best_fitness = float('inf')
        unchanged_generations = 0

        for _ in range(generations):
            selected_individuals = self.selection(population, selection_size)      
            new_generation = []
            
            # Ensure all parents are used for crossover
            for i in range(len(selected_individuals)):
                parent1 = selected_individuals[i]
                parent2 = random.choice(selected_individuals)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, mutation_rate)
                child2 = self.mutate(child2, mutation_rate)
                new_generation.append(child1)
                new_generation.append(child2)
            
            # If the new generation is larger than the population size, trim it
            if len(new_generation) > population_size:
                new_generation = new_generation[:population_size]
            

            # population = new_generation
            # thay thế các phần tử không tốt ở cuối bằng new_generation
            population = population[:len(population)-len(new_generation)] + new_generation

            # Find the best solution in this generation
            current_best_solution = min(population, key=self.fitness)
            current_best_fitness = self.fitness(current_best_solution)

            # Check if the best solution has changed
            if current_best_fitness < best_fitness:
                best_solution = current_best_solution
                best_fitness = current_best_fitness
                unchanged_generations = 0
            else:
                unchanged_generations += 1

            # If the best solution has not changed for 5 generations, break the loop
            if unchanged_generations >= max_unchanged_generations:
                break

        return best_solution


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


    def check_penalty(self,X1, X2, map):

        x1 = round(X1[0], 2)
        y1 = round(X1[1], 2)
        x2 = round(X2[0], 2)
        y2 = round(X2[1], 2)

        map = np.array(map)
        objs = np.hstack((np.where(map == 1)[0].reshape(-1, 1), np.where(map == 1)[1].reshape(-1, 1)))
        if map[int(y1)][int(x1)] == 1 or map[int(y2)][int(x2)] == 1:
            # print(x1, y1)
            # print(y1, x1, y2, x2)
            # print(map[int(y1)][int(x1)])
            # print(map[int(y2)][int(x2)])
            return False
        else:
            obj_tmp = []

            t1 = int(x1)
            t2 = int(x2)

            if t1 > t2:
                t1 = int(x2)
                t2 = int(x1)

            u1 = int(y1)
            u2 = int(y2)

            if (u1 > u2):
                u1 = int(y2)
                u2 = int(y1)

            for i in range(int(t1) - 1, int(t2) + 1):
                for j in range(int(u1) - 1, int(u2) + 1):
                    if map[j][i] == 1:
                        obj_tmp.append([i, j])
            if len(obj_tmp) == 0:
                return True
            if x1 == x2:
                y_low = y1
                y_high = y2
                if y1 > y2:
                    y_low = y2
                    y_high = y1

                for obj in obj_tmp:
                    if obj[0] <= x1 <= obj[0] + 1 and (y_low <= obj[1] <= y_high or y_low <= obj[1] + 1 <= y_high):
                        return False
            elif y1 == y2:
                x_low = x1
                x_high = x2
                if x1 > x2:
                    x_low = x2
                    x_high = x1
                for obj in obj_tmp:
                    if obj[1] <= y1 <= obj[1] + 1 and (x_low <= obj[0] <= x_high or x_low <= obj[0] + 1 <= x_high):
                        return False
            else:
                a = (y2 - y1) / (x2 - x1)
                b = y1 - a * x1

                # print(x1, x2, obj_tmp)
                for obj in obj_tmp:
                    pt_y_top = obj[1]
                    pt_y_bottom = obj[1] + 1
                    pt_x_left = obj[0]
                    pt_x_right = obj[0] + 1

                    x_intersect_top = (pt_y_top - b) / a
                    x_intersect_bottom = (pt_y_bottom - b) / a
                    y_intersect_left = a * pt_x_left + b
                    y_intersect_right = a * pt_x_right + b
                    x_low = x1
                    x_high = x2
                    if x1 > x2:
                        x_low = x2
                        x_high = x1
                    y_low = y1
                    y_high = y2
                    if y1 > y2:
                        y_low = y2
                        y_high = y1

                    if ((pt_x_left <= x_intersect_top <= pt_x_right and y_low <= pt_y_top <= y_high)
                            or (pt_x_left <= x_intersect_bottom <= pt_x_right and y_low <= pt_y_bottom <= y_high)
                            or (pt_y_top <= y_intersect_left <= pt_y_bottom and x_low <= pt_x_left <= x_high)
                            or (pt_y_top <= y_intersect_right <= pt_y_bottom and x_low <= pt_x_right <= x_high)):
                        # print(obj, x1, y1, x2, y2)
                        # print(x1, y1, x2, y2)
                        return False
                    else:
                        continue

        return True


    def a_star(self, start, goal):
        # Implement A* algorithm here
        pass

    def create_individual(self, start, goal):
        x, y = start
        individual = [(x, y)]
        visited_coordinates = set([(x, y)])

        dx, dy = goal[0] - start[0], goal[1] - start[1]
        steps = max(abs(dx), abs(dy))
        dx, dy = dx / steps, dy / steps

        last_direction = (dx, dy)

        for step in range(steps):
            x, y = round(x + dx), round(y + dy)
            if 0 <= x < len(self.grid) and 0 <= y < len(self.grid[0]) and self.grid[x][y] != 1:
                current_direction = (dx, dy)
                if current_direction != last_direction:
                    if self.check_penalty(individual[-1], (x, y), self.grid):
                        individual.append((x, y))
                        visited_coordinates.add((x, y))
                        last_direction = current_direction
            else:
                # If there is an obstacle, use A* to go around it
                path = self.a_star((x, y), goal)
                for point in path:
                    individual.append(point)
                    visited_coordinates.add(point)
                break

        # Add the goal to the path if it's not already there
        if individual[-1] != goal:
            if self.check_penalty(individual[-1], goal, self.grid):
                individual.append(goal)

        return individual
    def fitness(self, individual):
        if len(individual) == 0:
            return 9999
        return len(individual)

    def selection(self, population, selection_size):
        non_empty_arrays = [arr for arr in population if len(arr) > 0]
        selected_individuals = sorted(non_empty_arrays, key=lambda arr: self.fitness(arr) - random.random())[:selection_size]
        return selected_individuals

    def crossover(self, parent1, parent2):
        crossover_points = []
        for i in range(1, len(parent1)-1):
            for j in range (1, len(parent2)-1):
                if parent1[i] == parent2[j] and random.random() < 0.5:
                    crossover_points.append((i,j))

        if len(crossover_points) == 0:
            return parent1, parent2

        if len(crossover_points) == 1:
            child1 = parent1[:crossover_points[0][0]] + parent2[crossover_points[0][1]:]
            child2 = parent2[:crossover_points[0][1]] + parent1[crossover_points[0][0]:]
        else:
            child1 = parent1[:crossover_points[0][0]] + parent2[crossover_points[0][1]:crossover_points[-1][1]] + parent1[crossover_points[-1][0]:]
            child2 = parent2[:crossover_points[0][1]] + parent1[crossover_points[0][0]:crossover_points[-1][0]] + parent2[crossover_points[-1][1]:]

        return [child1, child2]

    def mutate(self, individual, mutation_rate=0.1):
        if random.random() < mutation_rate:
            mutation_points = [i for i in range(1, len(individual) - 1) if random.random() < mutation_rate]
            for mutation_point in mutation_points:
                X = individual[mutation_point]
                N = [(i, j) for i in range(len(self.grid)) for j in range(len(self.grid[0])) 
                     if self.grid[i][j] != 1 and (i, j) not in individual 
                     and (self.manhattan_distance((i, j), individual[mutation_point - 1]) == 1 
                          or self.manhattan_distance((i, j), individual[mutation_point + 1]) == 1)]
                while N:
                    Y = random.choice(N)
                    if self.manhattan_distance(Y, individual[mutation_point - 1]) == 1 and self.manhattan_distance(Y, individual[mutation_point + 1]) == 1:
                        individual[mutation_point] = Y
                        break
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

