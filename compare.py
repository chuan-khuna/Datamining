from LinearRegression import LinearRegression
from VisualizationLR import Visualization
import numpy as np
import matplotlib.pyplot as plt

num_sample = 50
# expected y = mx + c
m = 2
c = 5
min_x = -5
max_x = 10
x = np.linspace(min_x, max_x, num_sample)
y = m*x + c + np.random.randn(num_sample)
print("Expected y = {}x + {} + (random noise)".format(m, c))

learning_rates = [0.0001, 0.0002, 0.0005, 0.00075, 0.001, 0.0015]
max_iteration = 10
for alpha in learning_rates:
    lr = LinearRegression(x, y, learning_rate=alpha, max_iteration=max_iteration)
    vlr = Visualization(lr)
    vlr.cost_iteration()
    vlr.contour_cost()
    plt.show()