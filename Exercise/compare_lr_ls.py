import sys
sys.path.append("..")
import pandas as pd
import numpy as np
from LinearRegression import LinearRegression as LR
from VisualizationLR import Visualization as VLR
from LeastSquare import LeastSquare as LS
import matplotlib.pyplot as plt

num_sample = 100
m = 2
c = 5
min_x = -5
max_x = 10
x = np.linspace(min_x, max_x, num_sample)
y = m*x + c + np.random.randn(num_sample)
X = np.array([[1, i] for i in x])

print("Expected: y = {}x + {} + (random)".format(m, c))
ls = LS(X, y)
print("Least Square: h = {}x + {}".format(ls.parameters[1], ls.parameters[0]))


max_iteration = 1000
learning_rate = 0.0001
lr = LR(x, y, learning_rate=learning_rate, 
        max_iteration=max_iteration)
print("Linear Regression: h = {}x + {}".format(lr.parameters[-1][1], lr.parameters[-1][0]))

plt.plot(x, y, 'r.')
plt.plot(x, lr.predicted_y, 'b-')
plt.show()