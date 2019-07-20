import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd


grid_x = np.arange(-2.5, 2.5, 0.25)
grid_y = np.arange(-2.5, 2.5, 0.25)
X, Y = np.meshgrid(grid_x, grid_y)

def logistic(x, y, wx, wy, c):
    v = wx*x + wy*y + c
    return 1/(1 + np.exp(-v))

def gaussian(x, y, wx, wy, c, base):
    v = wx*x + wy*y + c
    return np.exp(-((v**2)/base**2))

def circle(x, y):
    v = x**2 + y**2
    return -v

wx = 0.4
wy = 0.4
c = -0.4
base = 0.2

# Z = gaussian(X, Y, wx, wy, c, base)
# Z = logistic(X, Y, 1, 1, -1.5)
Z = circle(X, Y)


fig = plt.figure()
ax = plt.axes(projection='3d')

plt.plot(grid_x, grid_y)

logistic_contour = ax.plot_surface(X, Y, Z, 
                rstride=1, cstride=1,
                cmap='viridis', edgecolor='none', alpha=0.75)


plt.plot([0, 1], [1, 0], 'rx')
plt.plot([0, 1], [0, 1], 'bx')


plt.xlabel('x')
plt.ylabel('y')
plt.show()