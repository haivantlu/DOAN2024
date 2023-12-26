import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from gui_map import *
import random as rd
from utils import *

# N_particles = 10
max_dimension = 10

def check_penalty(X1, X2, map):

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

# khoi tao x version 1

def init_x(map, start, goal):
    count = 0

    x_min = 0
    y_min = 0
    x_max = len(map) - 1
    y_max = len(map) - 1

    x_start, y_start = start
    x_goal, y_goal = goal

    points_set = [[y_start, x_start]]
    points_int = [y_start, x_start]

    while (True):
        o_x = rd.random()
        o_y = rd.random()

        px = points_set[-1][0] + 0.2 * o_x * (x_max - points_set[-1][0])
        py = points_set[-1][1] + 0.2 * o_y * (x_max - points_set[-1][1])

        if [int(px), int(py)] not in points_int:
            if check_penalty(points_set[-1], [px, py], map):
                points_set.append([px, py])
                points_int.append([int(px), int(py)])
                count = count + 1
                if count == max_dimension:
                    points_set.append([y_goal, x_goal])
                    break
                if int(px) == y_goal and int(py) == x_goal:
                    break
                else:
                    continue

    return points_set


# khoi tao x version 2

def init_x_v2(map, start, goal):

    x_start, y_start = start
    x_goal, y_goal = goal
    # x_start, y_start = x_start + 0.5, y_start + 0.5
    # x_goal, y_goal = x_goal + 0.5, y_goal + 0.5
    x_min = 1
    x_max = 5

    points_set = [(y_start, x_start)]
    points_int = [y_start, x_start]

    map_tmp = []
    for i in map:
        map_tmp.append(list(i))

    # print(check_penalty([y_goal + 0.01, x_goal + 1], [y_goal, x_goal], map))
    # print(check_penalty(points_set[-1], [y_goal, x_goal], map))
    # print(points_set[-1], [y_goal, x_goal])
    while not check_penalty(points_set[-1], [y_goal, x_goal], map):
        loop = 0
        while True:
            loop += 1
            x0 = (x_min + rd.random() * (x_max - x_min)) * (rd.randint(0, 2) - 1)
            y0 = (x_min + rd.random() * (x_max - x_min)) * (rd.randint(0, 2) - 1)

            x_curr = points_set[-1][0] + x0
            y_curr = points_set[-1][1] + y0
            x_curr, y_curr = check([x_curr, y_curr], points_set[-1], m=len(map), n=len(map))
            # print(points_set[-1], [x_curr, y_curr])
            # print(check_penalty(points_set[-1], [x_curr, y_curr], map))

            if not check_penalty([x_curr, y_curr], points_set[-1], map):
                x_curr = int(x_curr) + 0.5
                y_curr = int(y_curr) + 0.5
            if (check_penalty(points_set[-1], [x_curr, y_curr], map) and
                    map_tmp[int(x_curr)][int(y_curr)] == 0):
                # print(x_curr, y_curr)
                points_set.append([x_curr, y_curr])
                # print([x_curr, y_curr])
                map_tmp[int(x_curr)][int(y_curr)] = 3
                # i += 1
                break
            if loop > x_max * x_max * 40:
                # print(points_set[-1])
                points_set.pop()
                # print(0)
                # i -= 1
                break
        if len(points_set) == 0:
            # print(0)
            # i = 0
            # points_set.append([y_start, x_start])
            points_set = [[y_start, x_start]]
            map_tmp = []
            for i in map:
                map_tmp.append(list(i))
        # print(points_set[-1], [y_goal, x_goal])
    # print(1)
    points_set.append((y_goal, x_goal))
    return points_set


def simplify_path(x_init, map):
    x_tmp = [x_init[0]]
    l = len(x_init)
    pos = 0
    while not check_penalty(x_init[-1], x_tmp[-1], map):
        for i in range(l - 1, pos, -1):
            if check_penalty(x_init[i], x_tmp[-1], map):
                x_tmp.append(x_init[i])
                pos = i
                break
    x_tmp.append(x_init[-1])
    return x_tmp


def init_v(n_particles, n_points):
    return np.random.rand((n_particles, n_points - 1, 2)) * 3


def print_individual(file_name, point_set):
    l_dst, grid = read_file(file_name)
    obs = []
    n = len(grid)
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 1:
                obs.append([i, j])

    point_set = np.array(point_set)
    px = point_set[:, 0]
    py = point_set[:, 1]
    fig, ax = plt.subplots()

    ax.plot(px[0], py[0], marker='*')
    ax.plot(px[-1], py[-1], marker='*')
    ax.plot(px, py, color="blue")

    ax.plot([0, n, n, 0, 0], [0, 0, n, n, 0], color='red')

    for p in obs:
        a, b = p
        ax.add_patch(Rectangle((b, a), 1, 1, color="red"))

    for dst in l_dst:
        a, b = dst[0]-0.5, dst[1]-0.5
        ax.add_patch(Rectangle((b, a), 1, 1, color="yellow"))
    ax.invert_yaxis()
    plt.show()

# if __name__ == "__main__":
#     c = 0
#     G = GridMap(n_square=20, square_width=20, square_height=20)
#     grid = G.create_grid_map()
#     while(True):
#         tmp = []
#         for i in range(len(grid)):
#             for j in range(len(grid[i])):
#                 if grid[i][j] == 2:
#                     tmp.append(i)
#                     tmp.append(j)
#         start = [tmp[0], tmp[1]]
#         goal = [tmp[2], tmp[3]]
#
#         x_init = init_x_v2(grid, start, goal)
#         x_init_simplify = simplify_path(x_init, grid)
#         # print(x_init_simplify)
#         if (check_penalty(x_init_simplify[-2], x_init_simplify[-1], grid)):
#             print(x_init_simplify)
#             print_individual(grid, x_init_simplify)
#             c = c + 1
#             # break
#
#         if(c==10):
#             break
