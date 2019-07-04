import numpy as np
import matplotlib.pyplot as plt

class LinearClassification:
    def __init__(self, x, expected_y, initial_params, learning_rate=0.1, max_iteration=100):
        self.x = x
        self.expected_y = expected_y
        self.learning_rate = learning_rate
        self.max_iteration = max_iteration
        self.predicted_y = []
        self.costs = []
        self.parameters = [initial_params]
        self.linear_classification()

    def cal_predicted_class(self):
        """
            logistic function:
                1/ (1*exp(-x))
            use logistic function as activation function
            x >= 0 -->  class 1
            x < 0 --> class 0
            h(x) = logistic(x)
        """
        logistic = lambda x: 1/(1+np.exp(-x))
        weights = self.parameters[-1]
        v = (self.x).dot(weights.T)
        self.predicted_y = logistic(v)

    def cost_function(self):
        cost = (-self.expected_y).dot(np.log(self.predicted_y)) - (1-self.expected_y).dot(np.log(1-self.predicted_y))
        cost = cost/(len(self.expected_y))
        self.costs.append(np.round(cost, 4))

    def gradient_descent(self):
        old_weights = self.parameters[-1]
        new_weights = old_weights - (self.learning_rate/len(self.expected_y))*(
                (self.predicted_y - self.expected_y).dot(self.x)
            )
        self.parameters.append(np.around(new_weights[0], 4))
    
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
    and_in = np.array([
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ])
    and_init_w = np.array([0.5, 1.5, 1.5])
    and_out = np.array([[0, 0, 0, 1]])
    learning_rate = 0.5
    max_iteration = 10
    lc = LinearClassification(and_in, and_out, initial_params=and_init_w, learning_rate=learning_rate, max_iteration=max_iteration)
    x_axis = np.arange(-1, 2, 0.1)
    x1 = and_in[:, 1]
    x2 = and_in[:, 2]
    weights = lc.parameters[-1]
    w0 = weights[0]
    w1 = weights[1]
    w2 = weights[2]
    print(f"boundary = {w1}x1 + {w2}x2 + {w0}")
    print(f"cost: {lc.costs[-1]}")
    # visualize and
    plt.plot(x_axis, -(w1*x_axis + w0)/w2, label=f"boundary = {w1}$x_1$ + {w2}$x_2$ + {w0}")
    plt.plot(x1[:3], x2[:3], 'bo')
    plt.plot(x1[-1], x2[-1], "ro")
    plt.axis([-0.25, 1.25, -0.25, 1.25])
    plt.title(f"AND Problem, alpha: {learning_rate}, max iteration: {max_iteration}")
    plt.legend()
    plt.show()