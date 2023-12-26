import numpy as np
import math
from gui_map import *
from path_planning import *
from utils import *
#số cá thể
N_particles = 20


def pso_algorithm(file_name):
    # c = 0
    # x = []
    list_path = []
    l_dst, grid = read_file(file_name)
    print('he',l_dst)

    len_dst = len(l_dst)
    list_distance = [[[0, []] for _ in range(len_dst)] for _ in range(len_dst)]
    cnt = 0
    for p in range(len_dst - 1):
        idx = p
        for q in range(p + 1, len_dst):
            start = l_dst[idx]
            x = []
            cnt = cnt + 1
            goal = l_dst[q]
            c = 0
            # print(len(grid))
            while True:
                # tmp = []
                # for i in range(len(grid)):
                #     for j in range(len(grid[i])):
                #         if grid[i][j] == 2:
                #             tmp.append(i)
                #             tmp.append(j)
                # start = [tmp[0], tmp[1]]
                # goal = [tmp[2], tmp[3]]
                x_init = init_x_v2(grid, start, goal)
                x_init_simplify = simplify_path(x_init, grid)
                # print(x_init_simplify)
                # print_individual(grid, x_init_simplify)

                x.append(x_init_simplify)
                c += 1
                print("Done ca the: ", c)
                if c == N_particles:
                    break

            # print("Before: ", x)

            max_dimension_x = 0
            for i in x:
                max_dimension_x = max(max_dimension_x, len(i))

            x_after = []
            for i in range(N_particles):
                tmp = []
                for j in range(max_dimension_x):
                    if j < len(x[i]):
                        tmp.append(x[i][j])
                    else:
                        tmp.append(x[0][-1])
                x_after.append(tmp)

            x = np.array(x_after)
            # print("After: ", x)
            # print("Size: ", x.shape)

            if x.shape[1] == 2:
                x_tmp = x[0]
                kc = math.sqrt((x_tmp[0][0] - x_tmp[1][0]) ** 2 + (x_tmp[0][1] - x_tmp[1][1]) ** 2)
                print("Khoang cach: ", kc)
                list_distance[idx][q] = [kc, [start, goal]]
                # re_way = list(way)
                # re_way.reverse()
                list_distance[q][idx] = [kc, [goal, start]]
                continue

            start = x[0][0]
            goal = x[0][-1]

            x = x[:, 1: -1, :]
            # print("Size: ", x.shape)

            print(fitness(x, start, goal))

            # init x and v
            size, dim_0, dim_1 = x.shape[0], x.shape[1], x.shape[2]
            # v_s = init_v(size, dim_0)

            # Maximum iterations
            imax = 300

            # Acceleration coefficients
            c1 = 1.5
            c2 = 1.5

            # Weight
            wmax = 1.9
            wmin = 1.2

            # Debug level
            VERBOS = True

            # Max and min position bounds
            xmax = 4
            xmin = 1

            # Max and min velocity bounds
            vmax = 1   * (xmax - xmin)
            vmin = 0

            # Velocity
            v = np.zeros((size, dim_0, dim_1))
            v = vmin + ((vmax - vmin) * np.random.rand(size, dim_0, dim_1))

            # Fitness
            fx = np.zeros((size, 1))
            fx = fitness(x, start, goal)

            # Pbest
            pb = np.zeros((size, dim_0, dim_1))
            fpb = np.zeros((size, 1))
            pb = np.copy(x)
            fpb = np.copy(fx)

            # Gbest
            gb = np.zeros((1, dim_0, dim_1))
            fgb = 0
            gb = np.copy(x[np.argmin(fx), :].reshape(1, dim_0, dim_1))
            fgb = np.copy(fx[np.argmin(fx)]).reshape(1, 1)

            for i in range(imax):
                w = wmax - ((wmax - wmin) / imax) * i
                for k in range(size):

                    # Update partcile's velocity
                    v[k, :] = (w * v[k, :]) + (c1 * np.random.rand(1, dim_0, dim_1) * (pb[k, :] - x[k, :])) + (
                            c2 * np.random.rand(1, dim_0, dim_1) * (gb[0, :] - x[k, :]))

                    # Update particle's position
                    x_tmp = x[k, :]
                    v_tmp = v[k, :]
                    x[k, :] = x[k, :] + v[k, :]

                    x_ = [start]
                    check_pen = True
                    for p in x[k, :]:
                        f_x, f_y = check(p, x_[-1], m=len(grid), n=len(grid))
                        if check_penalty([f_x, f_y], x_[-1], grid):
                            x_.append([f_x, f_y])
                        else:
                            check_pen = False
                            x[k, :] = x_tmp
                            continue

                    f_x, f_y = check(goal, x_[-1], m=len(grid), n=len(grid))
                    if check_penalty([f_x, f_y], x_[-1], grid):
                        x_.append([f_x, f_y])
                    else:
                        check_pen = False

                    if check_pen:
                        x[k, :] = x_[1: -1]
                    else:
                        x[k, :] = x_tmp
                        continue

                    fx[k, 0] = fitness(x[k, :].reshape(1, dim_0, dim_1), start, goal)
                    # Update pbest
                    if fx[k, 0] < fpb[k, 0]:
                        pb[k, :] = x[k, :]
                        fpb[k, 0] = fx[k, 0]

                    # Update gbest
                    if fx[k, 0] < fgb:
                        gb[0, :] = x[k, :]
                        fgb = fx[k, 0]
                if VERBOS:
                    print(' Iteration', i, 'Global best', fgb)

            gb_ = [list(start)]

            gb = gb[0]
            for i in gb:
                gb_.append(list(i))

            gb_.append(list(goal))
            print(gb_)
            way = []
            for i in gb_:
                way.append([i[1], i[0]])
            # print_individual(grid, gb_)
            # list_path.append(gb_)c
            list_distance[idx][q] = [fgb, way]
            re_way = list(way)
            re_way.reverse()
            list_distance[q][idx] = [fgb, re_way]

            print("{}. Khoang cach tu diem {} den diem {} la: {}".format(cnt, idx, q, fgb))
    # for p in list_path:
    #     print_individual(file_name, p)
    return list_distance, grid, l_dst
# pso_algorithm()
