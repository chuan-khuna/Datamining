import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LinearRegression:
    def __init__(self, x, y, learning_rate=0.05, max_iteration=500):
        '''
            model:
                h = w0*x0 + w1*x1    ; x0 = 1, x1 = x from input
            cost funtion:
                cost = 0.5 * sum_square(predict - expected)/m       ; m = number of y
            gradient descent:
                new_weight = old_weight - learning_rate*(sum( d * xi ))     ; d = predict - expected
        '''
        self.x = x
        self.expected_y = y
        self.learning_rate = learning_rate
        self.max_iteration = max_iteration
        self.predicted_y = []
        self.costs = []
        self.parameters = [[1, 1]]          # initial parameter


    def cal_predicted_y(self):
        self.predicted_y = x*self.parameters[-1][1] + self.parameters[-1][0]

    def cal_cost(self):
        sum_square = np.sum(
            (self.predicted_y-self.expected_y)**2
        )
        cost = 0.5 * sum_square/len(self.expected_y)
        cost = round(cost, 4)
        self.costs.append(cost)
    
    def gradeint_desc(self):
        w0 = self.parameters[-1][0] - (self.learning_rate * (np.sum( (self.predicted_y - self.expected_y) )))
        w1 = self.parameters[-1][1] - (self.learning_rate * (np.sum( (self.predicted_y - self.expected_y)*x )))
        w0 = round(w0, 4)
        w1 = round(w1, 4)
        self.parameters.append([w0, w1])

    def linear_regression(self):
        for i in range(1, self.max_iteration+1):
            self.cal_predicted_y()
            self.cal_cost()
            self.gradeint_desc()
        self.cal_cost()     # calculate last iteration cost

    def plot_cost_iteration(self):
        iterations = np.arange(0, self.max_iteration+1, 1)
        plt.plot(iterations, self.costs, 'b.')
        plt.plot(iterations, self.costs)
        plt.grid(True, alpha=0.5)
        plt.xlabel("iteration")
        plt.ylabel("error cost")
        plt.show()


if __name__ == "__main__":
    num_point = 100
    max_iteration = 100
    x = np.linspace(0, 10, num_point)
    y = 2*x + 5

    lr = LinearRegression(x, y, learning_rate=0.0005, max_iteration=max_iteration)
    lr.linear_regression()
    lr.plot_cost_iteration()