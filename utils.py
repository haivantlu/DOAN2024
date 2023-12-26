import numpy as np
import math
from copy import deepcopy

def read_file(filename):
    with open(filename, "r") as fp:
        f_n = int(fp.readline())
        f_map = []
        for _ in range(f_n):
            s = fp.readline()
            s_l = s.strip().split(" ")
            f_map.append(list(map(int, s_l)))
    return f_map

def fitness(x, start, goal):
    size, dim_0, dim_1 = x.shape
    y = np.zeros((size, 1))

    for i, y_i in enumerate(x):
        tong = 0
        tong = tong + math.sqrt(
                    np.power((start[0] - y_i[0][0]), 2) + np.power((start[1] - y_i[0][1]), 2))
        tong = tong + math.sqrt(
                    np.power((goal[0] - y_i[-1][0]), 2) + np.power((goal[1] - y_i[-1][1]), 2))

        for j in range(len(y_i)):
            if j != len(y_i) - 1:
                tong = tong + math.sqrt(
                    np.power((y_i[j + 1][0] - y_i[j][0]), 2) + np.power((y_i[j + 1][1] - y_i[j][1]), 2))
        y[i] = tong

    return (y)


def check(point_curr, point_last, m, n):
    f_x, f_y = point_curr[0], point_curr[1]
    pre_x, pre_y = point_last[0], point_last[1]

    if f_x < 0:
        f_a = (f_y - pre_y) / (f_x - pre_x)
        f_b = f_y - f_a * f_x
        f_y = f_b
        f_x = 0
    elif f_x >= n:
        f_a = (f_y - pre_y) / (f_x - pre_x)
        f_b = f_y - f_a * f_x
        f_x = n - 0.01
        f_y = f_a * f_x + f_b
    if f_y < 0:
        f_a = (f_x - pre_x) / (f_y - pre_y)
        f_b = f_x - f_a * f_y
        f_x = f_b
        f_y = 0
    elif f_y >= m:
        f_a = (f_x - pre_x) / (f_y - pre_y)
        f_b = f_x - f_a * f_y
        f_y = m - 0.01
        f_x = f_a * f_y + f_b
    if f_x > n - 0.01:
        f_x = n - 0.01
    if f_y > m - 0.01:
        f_y = m - 0.01
        f_x = round(f_x, 2)
        f_y = round(f_y, 2)

    return round(f_x, 2), round(f_y, 2)

# print(read_file("data/map15_3.txt"))