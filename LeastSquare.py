import numpy as np

class LeastSquare:
    """
        Least Square:
            h = w0x0 + w1x1 + ....
            parameters = inverse((X.T) * X) * (X.T * y)   ; x*y = x dot y
            parameters = [w0 w1 w2 ...]

        X = [
            [x0 x1 ...]
            [x0 x1 ...]
            ...
        ]
        
        y = [
            y0
            y1
            ...
        ]
    """
    def __init__(self, x, y):
        self.x = x
        self.expected_y = y
        self.parameters = []
        self.least_square()

    def least_square(self):
        x_t = self.x.transpose()
        self.parameters = np.linalg.inv(
            x_t.dot(self.x)
        ).dot( x_t.dot(self.expected_y) )

        self.parameters = np.around(self.parameters, 4)

if __name__ == "__main__":
    # Example from NCC's book
    # x = np.array([
    #     [-1, 0, 0],
    #     [-1, 0, 1],
    #     [-1, 1, 0],
    #     [-1, 1, 1],
    # ])
    # y = np.array([
    #     -1, -1, -1, 1
    # ])

    # ls = LeastSquare(x, y)
    # print("AND Example from NCC's book",ls.parameters)

    num_sample = 100
    m = 2
    c = 5
    min_x = -5
    max_x = 10
    x = np.linspace(min_x, max_x, num_sample)
    y = m*x + c + np.random.randn(num_sample)
    X = np.array([[1, i] for i in x])
    ls = LeastSquare(X, y)
    print("Expected: y = {}x + {} + (random)".format(m, c))
    print("Least Square: h = {}x + {}".format(ls.parameters[1], ls.parameters[0]))