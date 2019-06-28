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

    def cost_function(self):
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
            self.cost_function()
            self.gradeint_desc()
        # calculate last iteration cost and predicted
        self.cal_predicted_y()
        self.cost_function()

    def plot_cost_iteration(self):
        iterations = np.arange(0, self.max_iteration+1, 1)
        plt.plot(iterations, self.costs, 'b.')
        plt.plot(iterations, self.costs)
        plt.title('Learning Rate: {}'.format(self.learning_rate))
        plt.xlabel("iteration")
        plt.ylabel("error cost")
        plt.grid(True, alpha=0.5)
        plt.show()

    def plot_scatter_model(self):
        plt.plot(self.x, self.expected_y, 'r.', label='expected')
        plt.plot(self.x, self.predicted_y, 'b-', label='predicted: h = {}x + {}'.format(round(self.parameters[-1][1], 3), round(self.parameters[-1][0], 3)))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.show()

    def contour_plot(self):
        # prepare data for contour plot
        parameters = np.array(self.parameters)
        w0 = parameters[:, 0]
        w1 = parameters[:, 1]
        w0_min,w0_max = np.amin(w0),np.amax(w0)
        w1_min,w1_max = np.amin(w1),np.amax(w1)
        costs = np.array(self.costs)
        # contour plot coordinate
        grid_res = 100
        x0 = np.linspace(w0_min/2, w0_max*2, grid_res)
        x1 = np.linspace(w1_min/2, w1_max*2, grid_res)
        X0, X1 = np.meshgrid(x0, x1)
        cost_grid = np.zeros(shape=(x0.size, x1.size))
        levels = np.unique(np.sort(costs))

        # calculate cost grid
        for i, v0 in enumerate(x0):
            for j, v1 in enumerate(x1):
                predicted = self.x*v1 + v0
                cost = np.sum((predicted - self.expected_y)**2)
                cost = round(0.5*cost/len(predicted), 1)
                cost_grid[i][j] = cost

        # contour line
        # contour = plt.contour(X0, X1, cost_grid, levels, cmap=plt.get_cmap('cool_r'), alpha=0.8)
        contour = plt.contourf(x0, x1, cost_grid, levels,)

        # plot parameters
        plt.scatter(w0, w1)
        plt.xlabel('w0')
        plt.ylabel('w1')
        plt.show()


if __name__ == "__main__":
    num_sample = 100
    max_iteration = 100
    learning_rate = 0.0001
    # y = mx + c
    m = 2       # slope
    c = 5       # constant
    min_x = 0
    max_x = 10
    x = np.linspace(min_x, max_x, num_sample)
    y = m*x + c + np.random.randn(num_sample)

    lr = LinearRegression(x, y, learning_rate=learning_rate, max_iteration=max_iteration)
    lr.linear_regression()
    lr.plot_cost_iteration()
    lr.plot_scatter_model()
    lr.contour_plot()
