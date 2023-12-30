from tkinter import Tk, Label, Button
from gui_map import GridMap
from pso_algorithm import *
from functools import partial
from GA import GA
from visualize import VisualizeResult
# from d import GeneticAlgorithm 
# from d_a_star import GeneticAlgorithm 
# from d_dfs import GeneticAlgorithm 
# from e_Van import GeneticAlgorithm 
from Beam_Search import GeneticAlgorithm 
# from e_Van_2 import GeneticAlgorithm 
# from dream import GeneticAlgorithm 
import time
from a import View 
from BAB import TravellingSalesman


def multi_goal(file_name):
    time_start = time.time()
 
  
    my_map = read_file(file_name)
    # print('hi',l_dst)
    ga_instance = GeneticAlgorithm(my_map)

    # ma trận đường đi
    distance_matrix, path_matrix, list_start = ga_instance.CalcMatrix()
    

    # GA
    # ga = GA(distance_matrix)
    # solve = ga.run()
    
    ga = GA(distance_matrix, pop_size=50, elite_size=30, mutation_rate=0.01, generations=50)
    best_distance, best_route = ga.run()
    print("Best Route: ", best_route)

    # BB
    # tsp = TravellingSalesman(distance_matrix)
    # solve = tsp.calculate_min_path()


    # Thứ tự đường đi
    # thuTuDuyet = solve[1]
    thuTuDuyet = best_route

    # lưu đường đi theo thứ tự
    list_path = []
    for i in range(len(thuTuDuyet)):
        if i == len(thuTuDuyet)-1:
            list_path.append(path_matrix[thuTuDuyet[i]][thuTuDuyet[0]])
        else:
            list_path.append(path_matrix[thuTuDuyet[i]][thuTuDuyet[i+1]])



    # total_distance = solve[0]
    total_distance = best_distance
    visualize = VisualizeResult(my_map, total_distance, list_path)

    # chuyển đường đi sang tọa độ
    # lưu đường đi theo thứ tự
    list_path = []
    for i in range(len(thuTuDuyet)):
            list_path.append(list_start[thuTuDuyet[i]])

        

    time_end = time.time()
    print("Quãng đường: ",total_distance)
    print("Đường đi", list_path)
    print("Thời gian chạy: ", time_end - time_start)

    visualize.showSolution()# biểu đồ 2


class MyFirstGUI:
    def __init__(self, master, file_name):
        self.master = master
        master.title("GUI")

        self.label = Label(master, text="Mobile Robot Path Planning")
        self.label.pack(pady = 10)
        self.label.config(font=("Times New Roman", 15))
        #click vào button
        self.create_map = Button(master, text="Solve", command=partial(multi_goal, file_name),
                                 height = 2, width = 10)

        self.create_map.pack(pady = 10)
        self.create_map.config(font=("Times New Roman", 15))

        self.close_button = Button(master, text="Close", command=master.quit,
                                   height = 2, width = 10)
        self.close_button.pack(pady = 10)
        self.close_button.config(font=("Times New Roman", 15))


# file_name = "data/GA và BB/map_20_5s_3.txt"
<<<<<<< HEAD
file_name = "data/TestCase1/map_20_5s_10.txt"
# file_name = "data/TestCase2/map_20_7s.txt"
=======
file_name = "data/TestCase2/map_20_23s.txt"
>>>>>>> 81684d9cdcdbf8769bf4672105d3b84a82038e7d

root = Tk()
root.geometry("310x240")


my_gui = MyFirstGUI(root, file_name)

root.mainloop()