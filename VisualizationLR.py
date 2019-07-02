import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from LinearRegression import LinearRegression

class Visualization:
    def __init__(self, linear_regression_object, extend_axis=1):
        self.lr = linear_regression_object
        self.x = np.array(self.lr.x)
        self.expected_y = np.array(self.lr.expected_y)
        self.predicted_y = np.array(self.lr.predicted_y)
        self.costs = np.array(self.lr.costs)
        self.parameters = np.array(self.lr.parameters)
        self.alpha = self.lr.learning_rate
        self.extend_axis = extend_axis

    def contour_grid_level_round(self, costs, option=5):
        if option == 5:
            # round in 5
            levels = np.unique(np.sort(np.round(costs/5, 0)*5))
        elif option == 0.25:
            levels = np.unique(np.sort(np.round(costs/25, 2)*25))
        else:
            levels = np.unique(np.sort(np.round(costs, option)))
        return levels

    def generate_contour_grid(self, grid_res=100):
        # model weight
        w0 = self.parameters[:, 0]
        w1 = self.parameters[:, 1]
        w0_min,w0_max = np.amin(w0), np.amax(w0)
        w1_min,w1_max = np.amin(w1), np.amax(w1)

        # contour plot coordinate
        x0 = np.linspace(w0_min-self.extend_axis, w0_max+self.extend_axis, grid_res)
        x1 = np.linspace(w1_min-self.extend_axis, w1_max+self.extend_axis, grid_res)
        costs = np.zeros(shape=(x0.size, x1.size))
        # calculate cost of grid
        for i, v1 in enumerate(x1):
            for j, v0 in enumerate(x0):
                predicted = self.x*v1 + v0
                sse = np.sum((predicted - self.expected_y)**2)
                costs[i][j] = round(0.5*sse/len(predicted), 4)

        return w0, w1, x0, x1, costs

    def animation_expected_predicted(self, delay=0.001):
        fig = plt.figure('Linear Regression Animation', figsize=(10, 5))
        plt.ion()

        for i, p in enumerate(self.parameters):
            plt.clf()
            expected = plt.plot(self.x, self.expected_y, 'r.', label='expected')
            w0 = p[0]
            w1 = p[1]
            x = np.linspace(np.amin(self.x) - self.extend_axis, np.amax(self.x) + self.extend_axis, 100)
            y = w1*x + w0
            predicted = plt.plot(x, y, 'b-', label='predicted: h = {}x + {}'.format(p[1], p[0]))
            plt.axis([np.amin(self.x) - self.extend_axis, np.amax(self.x) + self.extend_axis, 
                np.amin(self.expected_y) - self.extend_axis, np.amax(self.expected_y) + self.extend_axis
            ])
            plt.title("Alpha: {}, Iteration: {}, Cost: {}".format(
                self.alpha, i, self.costs[i]
            ))
            plt.legend()
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True, alpha=0.5)
            plt.draw()
            plt.pause(delay)
        plt.ioff()
        plt.show()

    def animation_cost_iteration(self, delay=0.001):
        fig = plt.figure("Cost-Iteration animation plot", figsize=(10, 5))
        plt.ion()
        for i, c in enumerate(self.costs):
            plt.plot(i, c, 'b.')
            plt.title("Alpha: {}, Iteration: {}, Cost: {}".format(
                self.alpha, i, self.costs[i]
            ))
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.grid(True, alpha=0.5)
            plt.draw()
            plt.pause(delay)
        plt.plot(np.arange(0, self.lr.max_iteration+1, 1), self.costs, 'b-')
        plt.ioff()
        plt.show()
        
    def animation_contour_cost(self, delay=0.001, grid_res=100, level_round=10):
        fig = plt.figure("Contour-Cost animation plot", figsize=(10, 5))
        plt.ion()
        w0, w1, x0, x1, costs = self.generate_contour_grid(grid_res=grid_res)
        levels = self.contour_grid_level_round(costs, option=level_round)

        # color map
        cm = plt.cm.get_cmap('Wistia')

        for i, c in enumerate(self.costs):
            plt.clf()
            contour = plt.contour(x0, x1, costs, levels,colors='black', linestyles='dashed', alpha=0.5)
            plt.clabel(contour, inline=1, fontsize=8)
            contour_bg = plt.contourf(x0, x1, costs, levels, cmap=cm)
            
            plt.plot(w0[:i], w1[:i], 'b.')
            plt.title("Alpha: {}, Iteration: {}, Cost: {}".format(
                    self.alpha, i, self.costs[i]
                ))
            plt.xlabel("w0")
            plt.ylabel("w1")
            plt.draw()
            plt.pause(delay)
        plt.plot(w0, w1, 'b')
        plt.plot(w0, w1, 'b.', label="Alpha: {}".format(self.alpha))
        plt.title("Alpha: {}, Iteration: {}, Cost: {}".format(self.alpha, len(self.costs)-1, self.costs[-1]))
        plt.ioff()
        plt.show()

    
    def expected_predicted(self):
        fig = plt.figure('Linear Regression', figsize=(10, 5))
        expected = plt.plot(self.x, self.expected_y, 'r.', label='expected')
        w0 = self.parameters[-1][0]
        w1 = self.parameters[-1][1]
        x = np.linspace(np.amin(self.x) - self.extend_axis, np.amax(self.x) + self.extend_axis, 100)
        y = w1*x + w0
        predicted = plt.plot(x, y, 'b-', label='predicted: h = {}x + {}'.format(w1, w0))

        plt.title("Alpha: {}, Iteration: {}, Cost: {}".format(
            self.alpha, len(self.parameters)-1, self.costs[-1]
        ))
        plt.axis([np.amin(self.x) - self.extend_axis, np.amax(self.x) + self.extend_axis, 
            np.amin(self.expected_y) - self.extend_axis, np.amax(self.expected_y) + self.extend_axis
        ])
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True, alpha=0.5)

    def cost_iteration(self):
        fig = plt.figure("Cost-Iteration plot", figsize=(10, 5))
        iterations = np.arange(0, self.lr.max_iteration+1, 1)
        plt.plot(iterations, self.costs, 'b.')
        plt.plot(iterations, self.costs, 'b-')
        plt.title("Alpha: {}".format(self.alpha))
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.grid(True, alpha=0.5)

    def pcolor_cost(self, grid_res=100, level_round=5):
        fig = plt.figure('Cost pcolor plot', figsize=(10, 5))

        w0, w1, x0, x1, costs = self.generate_contour_grid(grid_res=grid_res)
        levels = self.contour_grid_level_round(costs, option=level_round)

        # color map
        cm = plt.cm.get_cmap('Wistia')
        # pcolor -> bg color
        pcolor = plt.pcolormesh(x0, x1, costs, cmap=cm)
        plt.colorbar(pcolor, pad=0.01)
        plt.plot(w0, w1, 'b')
        plt.plot(w0, w1, 'b.', label="Alpha: {}".format(self.alpha))
        plt.title('Pcolor plot of Linear Regression & Gradient Descent'.format(self.alpha))
        plt.xlabel('w0')
        plt.ylabel('w1')
        plt.legend()

    def contour_cost(self, grid_res=100, level_round=5):
        fig = plt.figure('Cost contour plot', figsize=(10, 5))

        w0, w1, x0, x1, costs = self.generate_contour_grid(grid_res=grid_res)
        levels = self.contour_grid_level_round(costs, option=level_round)

        # color map
        cm = plt.cm.get_cmap('Wistia')

        contour = plt.contour(x0, x1, costs, levels,colors='black', linestyles='dashed', alpha=0.5)
        plt.clabel(contour, inline=1, fontsize=8)
        contour_bg = plt.contourf(x0, x1, costs, levels, cmap=cm)

        plt.plot(w0, w1, 'b')
        plt.plot(w0, w1, 'b.', label="Alpha: {}".format(self.alpha))
        plt.title('Contour plot of Linear Regression & Gradient Descent'.format(self.alpha))
        plt.xlabel('w0')
        plt.ylabel('w1')
        plt.legend()
    
    def show(self):
        plt.show()


if __name__ == "__main__":
    num_sample = 100
    max_iteration = 100
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
    print("Initial cost:", lr.costs[0])
    vlr = Visualization(lr)
    # vlr.animation_expected_predicted()
    # vlr.expected_predicted()
    # vlr.cost_iteration()
    # vlr.animation_contour_cost()
    vlr.contour_cost()
    vlr.show()