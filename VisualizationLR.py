import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from LinearRegression import LinearRegression

class Visualization:
    def __init__(self, linear_regression_object):
        self.lr = linear_regression_object
        self.x = np.array(self.lr.x)
        self.expected_y = np.array(self.lr.expected_y)
        self.predicted_y = np.array(self.lr.predicted_y)
        self.costs = np.array(self.lr.costs)
        self.parameters = np.array(self.lr.parameters)
        self.alpha = self.lr.learning_rate

    def animation(self, delay=0.01):
        fig = fig = plt.figure('Linear Regression Animation')
        plt.ion()

        for i, p in enumerate(self.parameters):
            plt.clf()
            expected = plt.plot(self.x, self.expected_y, 'r.', label='expected')
            w0 = p[0]
            w1 = p[1]
            extend = 1
            x = np.linspace(np.amin(self.x) - extend, np.amax(self.x) + extend, 100)
            y = w1*x + w0
            predicted = plt.plot(x, y, 'b-', label='predicted: h = {}x + {}'.format(p[1], p[0]))
            plt.axis([np.amin(self.x) - extend, np.amax(self.x) + extend, 
                np.amin(self.expected_y) - extend, np.amax(self.expected_y) + extend
            ])
            plt.title("Alpha: {}, Iteration: {}, Cost: {}".format(
                self.alpha, i, self.costs[i]
            ))
            plt.legend()
            plt.grid(True, alpha=0.5)
            plt .draw()
            plt.pause(delay)
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    num_sample = 100
    max_iteration = 1000
    learning_rate = 0.0001

    # expected y = mx + c
    m = 2
    c = 5
    min_x = -5
    max_x = 10
    x = np.linspace(min_x, max_x, num_sample)
    y = m*x + c + np.random.randn(num_sample)
    print("Expected y = {}x + {} + (random noise)".format(m, c))

    # Linear Regression
    lr = LinearRegression(x, y, learning_rate=learning_rate, max_iteration=max_iteration)
    print("Hypothesis y = {}x + {}".format(lr.parameters[-1][1], lr.parameters[-1][0]))
    print("Cost: {}".format(lr.costs[-1]))

    vlr = Visualization(lr)
    vlr.animation()