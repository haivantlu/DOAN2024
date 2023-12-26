from tkinter import Tk, Label, Button
from gui_map import GridMap
from pso_algorithm import *
from functools import partial
from GA import GA
from BAB import BAB
from visualize import VisualizeResult
import time




def multi_goal(file_name):
    time_start = time.time()
    list_distance, grid, l_dst = pso_algorithm(file_name)
    # print(list_distance)

    tsp_map = []

    for i in range(len(list_distance)):
        array = []
        for j in range(len(list_distance[i])):
            array.append(list_distance[i][j][0])
            # print(list_distance[i][j][0])
        tsp_map.append(array)

    bb = BAB(tsp_map)
    bb.solve()

    print("Quãng đường: ", bb.final_res)
    print("Đường đi", bb.final_path)

    visualize = VisualizeResult(grid, l_dst, list_distance, bb.final_path, bb.final_res)
    time_end = time.time()

    print("Thời gian chạy: ", time_end - time_start)
    visualize.showSolution()
    # vẽ
    # visualize.showSolutionDynamic()



class MyFirstGUI:
    def __init__(self, master, file_name):
        self.master = master
        master.title("GUI")

        self.label = Label(master, text="Mobile Robot Path Planning")
        self.label.pack(pady=10)
        self.label.config(font=("Times New Roman", 15))

        self.create_map = Button(master, text="Solve", command=partial(multi_goal, file_name),
                                 height=2, width=10)

        self.create_map.pack(pady=10)
        self.create_map.config(font=("Times New Roman", 15))

        self.close_button = Button(master, text="Close", command=master.quit,
                                   height=2, width=10)
        self.close_button.pack(pady=10)
        self.close_button.config(font=("Times New Roman", 15))



file_name = "data/GA và BB/map15_4.txt"

root = Tk()
root.geometry("310x240")
my_gui = MyFirstGUI(root, file_name)

root.mainloop()
