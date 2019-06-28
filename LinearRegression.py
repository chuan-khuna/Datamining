import numpy as np

class LinearRegression:
    def __init__(self, x, expected_y, learning_rate, max_iteration=500, initial_params=[0, 0]):
        self.x = x
        self.expected_y = expected_y
        self.learning_rate = learning_rate
        self.max_iteration = max_iteration
        self.predicted_y = []
        self.costs = []
        self.parameters = [initial_params]
        self.linear_regression()

    def cal_predicted_y(self):
        """
            calculate predicted y  ==> h(x)

            model (hypothesis):
                h = w1*x1  + w0*x0      ; x0 = 1, x1 = input x
                y = mx + c
        """
        self.predicted_y = (self.x * self.parameters[-1][1]) + self.parameters[-1][0]
    
    def cost_function(self):
        """
            calculate error cost of current model

            cost = 1/2 * sum_square( predicted - expected )/m       ; m = number of sample
        """
        
        sum_square = np.sum(
            (self.predicted_y-self.expected_y)**2
        )
        cost = 0.5 * sum_square/len(self.expected_y)
        self.costs.append(round(cost, 4))

    def gradient_descent(self):
        """
            adjust parameter by gradient descent method

            new_weight = old_weight - learning_rate * ( sum(d * xi) )       ; d = predicted - expected
        """
        w0 = self.parameters[-1][0] - self.learning_rate * np.sum(
                (self.predicted_y - self.expected_y) 
            )
        w1 = self.parameters[-1][1] - self.learning_rate * np.sum(
                (self.predicted_y - self.expected_y)*x 
            )
        w0 = round(w0, 4)
        w1 = round(w1, 4)
        self.parameters.append([w0, w1])
    
    def linear_regression(self):
        for i in range(1, self.max_iteration+1):
                    self.cal_predicted_y()      # calculate predicted y of current parameter
                    self.cost_function()        # calculate cost of current parameter
                    self.gradient_descent()        # adjust parameter
        # calculate last iteration predicted y and cost
        self.cal_predicted_y()
        self.cost_function()


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
    print("expected y = {}x + {} + (random noise)".format(m, c))

    # Linear Regression
    lr = LinearRegression(x, y, learning_rate=learning_rate, max_iteration=max_iteration)
    print("hypothesis y = {}x + {}".format(lr.parameters[-1][1], lr.parameters[-1][0]))
    print("cost: {}".format(lr.costs[-1]))