import matplotlib.pyplot as plt
import numpy as np


class View:

    def __init__(self, matrix):
        self.matrix = matrix
        
        # Tạo plot và hiển thị ma trận với màu sắc
        fig, ax = plt.subplots()
        cax = ax.matshow(self.matrix, cmap=plt.get_cmap('viridis'))


        # Hiển thị grid lines
        ax.set_xticks(np.arange(-.5, len(self.matrix[0]), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(self.matrix), 1), minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=2)

        plt.show()
