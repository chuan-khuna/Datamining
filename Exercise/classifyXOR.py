import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd

class XORClassify:
    def __init__(self, x, expected_y, initial_params, base=0.2, learning_rate=0.1, max_iteration=100):
        self.x = x
        self.expected_y = expected_y
        self.learning_rate = learning_rate
        self.max_iteration = max_iteration
        self.predicted_y = []
        self.costs = []
        self.parameters = [initial_params]
        self.base = base
        self.linear_classification()

    def cal_predicted_class(self):
        gaussian = lambda x: np.exp( -(x**2)/(2*self.base**2) )
        weights = self.parameters[-1]
        v = (self.x).dot(weights.T)
        self.predicted_y = gaussian(v)

    def cost_function(self):
        cost = (-self.expected_y).dot(np.log(self.predicted_y)) - (1-self.expected_y).dot(np.log(1-self.predicted_y))
        cost = cost/(len(self.expected_y))
        self.costs.append(np.round(cost, 4))

    def gradient_descent(self):
        old_weights = self.parameters[-1]
        new_weights = old_weights + (self.learning_rate/len(self.expected_y))*(
                (self.x).T.dot(self.predicted_y - self.expected_y)
            )
        self.parameters.append(np.around(new_weights, 4))
    
    def linear_classification(self):
        for i in range(1, self.max_iteration+1):
            self.cal_predicted_class()      # calculate model class using logistic function
            self.cost_function()            # calculate cost of current iteration parameters
            self.gradient_descent()        # adjust parameter
        # calculate last iteration predicted y and cost
        self.cal_predicted_class()
        self.cost_function()

if __name__ == "__main__":
    # and problem
    xor_in = np.array([
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ])
    xor_init_w = np.array([0.5, 1, 1])
    xor_out = np.array([0, 1, 1, 0])
    learning_rate = 0.1
    max_iteration = 200
    base = 0.1
    lc = XORClassify(xor_in, xor_out, 
        initial_params=xor_init_w, 
        learning_rate=learning_rate, 
        max_iteration=max_iteration,
        base=base
        )

    def gaussian(x, y, wx, wy, c, base):
        v = wx*x + wy*y + c
        return np.exp(-((v**2)/(base**2)))

    wx = lc.parameters[-1][1]
    wy = lc.parameters[-1][2]
    c = lc.parameters[-1][0]

    grid_x = np.arange(-0.5, 1.5, 0.1)
    grid_y = np.arange(-0.5, 1.5, 0.1)
    X, Y = np.meshgrid(grid_x, grid_y)
    Z = gaussian(X, Y, wx, wy, c, base)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # plt.plot(grid_x, grid_y)

    surface = ax.plot_surface(X, Y, Z, 
            rstride=1, cstride=1,
            cmap='plasma', edgecolor='none', alpha=1)
    
    plt.plot([0, 1], [1, 0], 'bo', markersize=10, label="class 1 True")
    plt.plot([0, 1], [0, 1], 'rX', markersize=10, label="class 0 False")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    print(f"weight: {lc.parameters[-1]}")
    print(f"cost: {lc.costs[-1]}")
    print(f"predicted: {np.round(lc.predicted_y, 2)}")
    plt.show()
