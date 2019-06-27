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
        plt.xlabel("iteration")
        plt.ylabel("error cost")
        plt.grid(True, alpha=0.5)
        plt.show()

    def plot_scatter_model(self):
        plt.plot(self.x, self.expected_y, 'r.', label='expected')
        plt.plot(self.x, self.predicted_y, 'b-', label='predicted')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.show()


if __name__ == "__main__":
    num_sample = 100
    min_x = 0
    max_x = 10
    max_iteration = 500
    learning_rate = 0.0001

    x = np.linspace(min_x, max_x, num_sample)
    y = 2*x + 5

    lr = LinearRegression(x, y, learning_rate=learning_rate, max_iteration=max_iteration)
    lr.linear_regression()
    lr.plot_cost_iteration()
    lr.plot_scatter_model()