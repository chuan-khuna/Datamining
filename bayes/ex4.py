import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd
import scipy as sp
import bayes_multi_attr as b
from sympy import *

sns.set_style("whitegrid")
sns.set_palette("bright")
sns.set_context('notebook', font_scale=1.25, rc={"lines.linewidth": 3})

c_map = 'rainbow'

x, y = np.mgrid[-10:10:0.1, -10:10:0.1]

# stack to (x, y) pair
xy = np.dstack([x, y])

# c1
mu1 = [3, 2]
sigma1 = [[1, 0], [0, 0.5]]
z1 = sp.stats.multivariate_normal.pdf(xy, mean=mu1, cov=sigma1)
prob_c1 = 0.5

# c2
mu2 = [-2, 0]
sigma2 = sigma1
z2 = sp.stats.multivariate_normal.pdf(xy, mean=mu2, cov=sigma2)
prob_c2 = 0.5

mat_mu1 = Matrix(mu1).T
mat_mu2 = Matrix(mu2).T
mat_sigma = Matrix(sigma1)
x1, x2 = symbols('x1 x2')
mat_x = matrix_x = Matrix([x1, x2]).T

b.surface_plot(x, y, [z1, z2])
b.contour_plot(x, y, [z1, z2])

g = b.g_lin(mat_mu1, mat_mu2, mat_sigma, mat_x, 0.5, 0.5)

print('g =', g)
g = solve(g, x2)
lamb_g = lambdify(x1, g, 'numpy')

x_val = np.arange(-10, 10, 0.001)
y_val = lamb_g(x_val)
plt.plot(x_val, y_val[0], 'b')
plt.contour(x, y, z1)
plt.contour(x, y, z2)
plt.contourf(x, y, z1/(z1+z2), alpha=0.75)
plt.show()
