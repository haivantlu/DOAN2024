import matplotlib.pyplot as plt
import numpy as np
import pygame
import math


class VisualizeResult:

    def __init__(self, grid, total_distance, path_matrix):
        self.grid = grid
        self.total_distance = total_distance
        self.path_matrix = path_matrix
        self.n = len(grid)
        self.solution = self.get_solution()
        self.distance = self.get_total_distance()
        self.destination = []

    def get_solution(self):
        # Flatten the path_matrix into a 1D list
        solution = [point for path in self.path_matrix for point in path]
        # print (solution)
        return solution

    def get_total_distance(self):
        # Calculate the total distance
  
        return round(self.total_distance, 3)

    def draw_grid(self, ax):
        for i in range(self.n):
            ax.plot([0, self.n], [i, i], color='black', linewidth='0.5')
            ax.plot([i, i], [0, self.n], color='black', linewidth='0.5')

    def draw_border(self, ax):
        ax.plot([0, self.n, self.n, 0, 0], [0, 0, self.n, self.n, 0], color='black')

    def draw_obstacles(self, ax, x_plt):
        for i_map in range(self.n):
            for j_map in range(self.n):
                if self.grid[i_map][j_map] == 1:
                    ax.fill_between(x_plt, self.n - 1 - i_map, self.n - i_map,
                                    where=(x_plt >= j_map) & (x_plt <= j_map + 1), color='black')

    def draw_destinations(self, ax):
        count = 0
        for i_map in range(self.n):
            for j_map in range(self.n):
                if self.grid[i_map][j_map] == 2:
                    self.destination.append((i_map, j_map))
                    color = 'green' if count == 0 else 'yellow'
                    ax.fill_between([j_map, j_map+1], self.n - 1 - i_map, self.n - i_map, color=color)
                    count += 1
    def draw_path(self, ax):
        prev_direction = None
        segment_start = 0
        for i in range(len(self.solution) - 1):
            i_map1, j_map1 = self.solution[i][0] + 0.5, self.solution[i][1] + 0.5
            i_map2, j_map2 = self.solution[i + 1][0] + 0.5, self.solution[i + 1][1] + 0.5
            dx = j_map2 - j_map1
            dy = (self.n - i_map2) - (self.n - i_map1)

            # Determine the direction of the current cell
            current_direction = (dx, dy)

            # If the direction has changed, draw an arrow from the start to the middle of the previous segment
            if current_direction != prev_direction and i > 0:
                mid_point = self.solution[segment_start + (i - segment_start) // 2]
                i_mid, j_mid = mid_point[0] + 0.5, mid_point[1] + 0.5
                dx_mid = j_mid - j_map1
                dy_mid = (self.n - i_mid) - (self.n - i_map1)
                ax.arrow(j_map1, self.n - i_map1, dx_mid, dy_mid, color='red', head_width=0.2, head_length=0.2, fc='red', ec='red')
                segment_start = i

            # Draw a line for the current cell
            ax.plot([j_map1, j_map2], [self.n - i_map1, self.n - i_map2], color='red')

            # Update the previous direction
            prev_direction = current_direction
                
    def showEnvironment(self):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=120)
        ax.set_title("Distance = " + str(self.distance))
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_xticklabels([])

        self.draw_grid(ax)
        self.draw_border(ax)
        self.draw_obstacles(ax, np.arange(0, self.n, 0.1))
        self.draw_destinations(ax)
        plt.show()

    def showSolution(self):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=120)
        ax.set_title("Distance = {}".format(self.distance))
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_xticklabels([])

        self.draw_grid(ax)
        self.draw_border(ax)
        self.draw_obstacles(ax, np.arange(0, self.n, 0.1))
        self.draw_destinations(ax)
        self.draw_path(ax)
        plt.show()

    # ... rest of your methods ...

    # ... rest of your methods ...

    def showSolutionDynamic(self):
        origin = 0.1

        list_des_pass = []
        list_des_arrive = []
        for i in self.solution:
            list_des_arrive.append([self.list_des[i][0], self.list_des[i][1]])
        start = list_des_arrive[0]
        list_des_arrive.append(start)

        pygame.init()
        scale = 600 / self.n
        win = pygame.display.set_mode((self.n * scale, self.n * scale))

        pygame.display.set_caption("Vân GA")
        list_move_py = []
        for i in self.list_move:
            list_move_py.append([i[1] * scale, i[0] * scale])
        x = list_move_py[0][0]
        y = list_move_py[0][1]

        width = 2 * origin * scale
        height = 2 * origin * scale

        vel = 1.7
        run = True
# hien thi duong di
        while run:
            pygame.time.delay(10)
            if len(list_move_py) >= 2:
                f_st = list_move_py[0]
                f_dst = list_move_py[1]
                if f_st[0] == f_dst[0] and f_st[1] == f_dst[1]:
                    list_move_py.pop(0)
                    f_st = list_move_py[0]
                    f_dst = list_move_py[1]
                f_h = math.sqrt((f_dst[1] - f_st[1]) * (f_dst[1] - f_st[1])
                                + (f_dst[0] - f_st[0]) * (f_dst[0] - f_st[0]))
                pi_x = (f_dst[0] - f_st[0]) / f_h
                print('pi_x', pi_x)
                pi_y = (f_dst[1] - f_st[1]) / f_h
                print('pi_y', pi_y)
                vel_x = vel * pi_x
                vel_y = vel * pi_y
                x += vel_x
                y += vel_y
                if (x - f_dst[0]) * (x - f_st[0]) > 0 or (y - f_dst[1]) * (y - f_st[1]) > 0:
                    x = f_dst[0]
                    y = f_dst[1]
                    if list_des_arrive[0][0] == round(y / scale, 2) and list_des_arrive[0][1] == round(x / scale, 2):
                        list_des_pass.append(list_des_arrive[0])
                        list_des_arrive.pop(0)
                    list_move_py.pop(0)
            else:
                x = list_move_py[0][0]
                y = list_move_py[0][1]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

            win.fill((255, 255, 255))
            pygame.draw.line(win, (255, 0, 0), (list_move_py[0][0], list_move_py[0][1]), (x, y))
            for des in list_des_arrive:
                pygame.draw.rect(win, (255, 255, 0), [(des[1] - 0.5) * scale, (des[0] - 0.5) * scale, scale, scale])
            for des in list_des_pass:
                pygame.draw.rect(win, (0, 255, 255), [(des[1] - 0.5) * scale, (des[0] - 0.5) * scale, scale, scale])

            pygame.draw.rect(win, (0, 0, 255), [x - origin * scale, y - origin * scale, width, height])

            for i in range(self.n):
                for j in range(self.n):
                    if self.environment[i][j] == 1:
                        pygame.draw.rect(win, (255, 0, 0), [j * scale, i * scale, scale, scale])
                        # pygame.draw.lines(win, (255, 0, 0), True, [[list_move_py[0][0], list_move_py[0][1]], [x, y]])

                        # pygame.draw.line(win, (255, 0, 0), [j * scale, i * scale, scale, scale])

            # tên lửa
            # pygame.draw.aaline(win, (255, 0, 0), (list_move_py[0][0], list_move_py[0][1]), (x, y))


            pygame.display.update()

        pygame.quit()
