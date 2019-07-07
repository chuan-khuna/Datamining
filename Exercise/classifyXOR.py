import numpy as np
import matplotlib.pyplot as plt

class XORClassify:
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
        gaussian = lambda x: np.exp(-((x)**2))
        weights = self.parameters[-1]
        v = (self.x).dot(weights.T)
        self.predicted_y = gaussian(v)

    def cost_function(self):
        cost = np.sum(
            (self.predicted_y - self.expected_y)**2
        )
        self.costs.append(np.round(cost, 4))

    def gradient_descent(self):
        old_weights = self.parameters[-1]
        new_weights = old_weights - (self.learning_rate/len(self.expected_y))*(
                (self.predicted_y - self.expected_y).dot(self.x)
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
    xor_init_w = np.array([0, 0, 0])
    xor_out = np.array([0, 1, 1, 0])
    learning_rate = 0.001
    max_iteration = 100
    lc = XORClassify(xor_in, xor_out, 
        initial_params=xor_init_w, 
        learning_rate=learning_rate, 
        max_iteration=max_iteration)

    print(f"weight: {lc.parameters[-1]}")
    print(f"cost: {lc.costs[-1]}")
    print(f"predicted: {np.round(lc.predicted_y, 2)}")
