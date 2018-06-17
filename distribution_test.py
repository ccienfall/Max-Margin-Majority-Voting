import numpy as np
import scipy
import os
import scipy.io as sio
import matplotlib.pyplot as plt

class dist_plot(object):
    def __init__(self):
        self.name = "dist_plot"
        self.data = None
    def get_dirichlet(self, prob):
        data = np.random.dirichlet(prob,10000)
        self.data = data

    def plot(self):
        plt.scatter(self.data[:,0], self.data[:,1],marker="+" )
        plt.show()

if __name__ == "__main__":
    D = dist_plot()
    D.get_dirichlet([1,1,1])
    print(D.data)
    D.plot()