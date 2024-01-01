from tkinter import Tk, Label, Button
from functools import partial
from visualize import VisualizeResult 
from GA_distance import GeneticAlgorithm 
import time
from utils import *
from BB import TravelingSalesman


def multi_goal(file_name):
    time_start = time.time()
    my_map = read_file(file_name)
    ga_instance = GeneticAlgorithm(my_map)

    # ma trận đường đi
    distance_matrix, path_matrix, list_start = ga_instance.CalcMatrix()
    
    # BB
    tsp = TravelingSalesman(distance_matrix)
    solve = tsp.calculate_min_path()

    # Thứ tự đường đi
    thuTuDuyet = solve[1]

    # lưu đường đi theo thứ tự
    list_path = []
    for i in range(len(thuTuDuyet)):
        if i == len(thuTuDuyet)-1:
            list_path.append(path_matrix[thuTuDuyet[i]][thuTuDuyet[0]])
        else:
            list_path.append(path_matrix[thuTuDuyet[i]][thuTuDuyet[i+1]])


    total_distance = solve[0]
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

        self.label = Label(master, text="GA-BB")
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

file_name = "data/TestCase5/map_15_19s.txt"

root = Tk()
root.geometry("310x240")
my_gui = MyFirstGUI(root, file_name)
root.mainloop()