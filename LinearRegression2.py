import numpy as np

class LinearRegression2:
    def __init__(self, x, expected_y, initial_params, learning_rate=0.00001, max_iteration=500):
        """
            xi = [x0 x1 ... xn]         ; n = number of attribute of x[i]
            x = [x1 x2 ..... xm]        ; m = number of sample
            predicted yi = xi * wi
        """
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
            Calculate predicted y  ==> h(x)

            model (hypothesis):
                h = wn*xn... + w1*x1  + w0*x0      ; x0 = 1
        """
        self.predicted_y = (self.x).dot(self.parameters[-1])
    
    def cost_function(self):
        """
            Calculate error cost of current model

            cost = 1/2 * sum_square( predicted - expected )/m       ; m = number of sample
        """
        sum_square = np.sum(
            (self.predicted_y-self.expected_y)**2
        )
        cost = 0.5 * sum_square/len(self.expected_y)
        self.costs.append(round(cost, 4))

    def gradient_descent(self):
        """
            Adjust parameter by gradient descent method

            new_weight = old_weight - learning_rate * ( sum(d * xi) )       ; d = predicted - expected
        """
        old_weights = self.parameters[-1]
        new_weights = old_weights - self.learning_rate * (self.predicted_y - self.expected_y).dot(self.x)

        self.parameters.append(np.around(new_weights, 4))
    
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
    X = np.array([[1, i] for i in x])

    Y = X.dot(np.array([c, m])) + np.random.randn(num_sample)
    # or y = m*x + c + np.random.randn(num_sample)
    # or y = m*X[:, 1] + X[:, 0]*c + np.random.randn(num_sample)

    print("Expected y = {}x + {} + (random noise)".format(m, c))
    initial_params = np.ones(X.shape[1])
    # Linear Regression
    lr = LinearRegression2(X, Y, initial_params=initial_params, learning_rate=learning_rate, max_iteration=max_iteration)
    print("Hypothesis y = {}x + {}".format(lr.parameters[-1][1], lr.parameters[-1][0]))
    print("Cost: {}".format(lr.costs[-1]))