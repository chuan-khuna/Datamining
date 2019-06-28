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
        fig = plt.figure('cost-iteration plot')
        iterations = np.arange(0, self.max_iteration+1, 1)
        plt.plot(iterations, self.costs, 'b.')
        plt.plot(iterations, self.costs)
        plt.title('Learning Rate: {}'.format(self.learning_rate))
        plt.xlabel("iteration")
        plt.ylabel("error cost")
        plt.grid(True, alpha=0.5)

    def plot_scatter_model(self):
        fig = plt.figure('predicted vs expercted plot')
        plt.plot(self.x, self.expected_y, 'r.', label='expected')
        plt.plot(self.x, self.predicted_y, 'b-', label='predicted: h = {}x + {}'.format(round(self.parameters[-1][1], 3), round(self.parameters[-1][0], 3)))
        plt.title('Predicted vs Expected')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.5)

    def pcolor_plot(self, grid_res=100):
        fig = plt.figure('cost contour plot')
        # prepare data for contour plot
        parameters = np.array(self.parameters)
        w0 = parameters[:, 0]
        w1 = parameters[:, 1]
        w0_min,w0_max = np.amin(w0),np.amax(w0)
        w1_min,w1_max = np.amin(w1),np.amax(w1)
        W0, W1 = np.meshgrid(w0, w1)
        
        # contour plot coordinate
        x0 = np.linspace(w0_min-1, w0_max+1, grid_res)
        x1 = np.linspace(w1_min-1, w1_max+1, grid_res)
        X0, X1 = np.meshgrid(x0, x1)
        costs = np.zeros(shape=(x0.size, x1.size))
        # calculate cost grid
        for i, v0 in enumerate(x0):
            for j, v1 in enumerate(x1):
                predicted = self.x*v1 + v0
                cost = np.sum((predicted - self.expected_y)**2)
                costs[i][j] = round(0.5*cost/len(predicted), 1)
        levels = np.unique(np.sort(costs))

        # color
        cm = plt.cm.get_cmap('plasma')
        # pcolor
        cm = plt.cm.get_cmap('plasma')
        # bg color
        contour = plt.pcolor(X0, X1, costs, cmap=cm)
        plt.colorbar(pad=0.01)

        # plot parameters
        plt.plot(w0, w1, 'r')
        plt.plot(w0, w1, 'r.', label="learning rate: {}".format(self.learning_rate))
        plt.title('Contour plot of Linear Regression & Gradient Descent'.format(self.learning_rate))
        plt.xlabel('w0')
        plt.ylabel('w1')
        plt.legend()

    def animation_predicted_plot(self):
        fig = plt.figure('Animation')
        plt.ion()
        for i, p in enumerate(self.parameters):
            plt.clf()
            scatter = plt.plot(self.x, self.expected_y, 'r.', label='expected')
            w0 = p[0]
            w1 = p[1]
            x = np.linspace(np.amin(self.x), np.amax(self.x), 100)
            y = w1*x + w0
            predicted_line = plt.plot(x, y, 'b-', label='predicted: h = {}x + {}'.format(p[1], p[0]))
            plt.axis([np.amin(self.x)-0.5, np.amax(self.x)+0.5, np.amin(self.expected_y)-0.5, np.amax(self.expected_y)+0.5])
            plt.title('iteration: {}, cost: {}'.format(i, self.costs[i]))
            plt.legend()
            plt.grid(True, alpha=0.5)
            plt.draw()
            plt.pause(0.001)
        plt.ioff() 
        plt.show()

if __name__ == "__main__":
    num_sample = 100
    max_iteration = 100
    learning_rate = 0.0005
    # y = mx + c
    m = 2       # slope
    c = 5       # constant
    min_x = 0
    max_x = 10
    x = np.linspace(min_x, max_x, num_sample)
    y = m*x + c + np.random.randn(num_sample)

    lr = LinearRegression(x, y, learning_rate=learning_rate, max_iteration=max_iteration)
    lr.linear_regression()
    lr.animation_predicted_plot()
    lr.plot_cost_iteration()
    lr.plot_scatter_model()
    lr.pcolor_plot()
    plt.show()