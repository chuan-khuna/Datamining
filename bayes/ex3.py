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
mu1 = [0, 5]
sigma1 = [[1, 0], [0, 2]]
z1 = sp.stats.multivariate_normal.pdf(xy, mean=mu1, cov=sigma1)
prob_c1 = 0.5

# c2
mu2 = [-3, 3]
sigma2 = [[2, 0], [0, 2]]
z2 = sp.stats.multivariate_normal.pdf(xy, mean=mu2, cov=sigma2)
prob_c2 = 0.5

b.surface_plot(x, y, [z1, z2])
b.contour_plot(x, y, [z1, z2])

x1, x2 = symbols('x1 x2')
matrix_x = Matrix([x1, x2]).T
matrix_mu1 = Matrix(mu1).T
matrix_sigma1 = Matrix(sigma1)
matrix_mu2 = Matrix(mu2).T
matrix_sigma2 = Matrix(sigma2)
# posterior = z1/(z1+z2)

g1 = b.g_i(1, matrix_x, matrix_mu1, matrix_sigma1, prob_c=prob_c1)
g2 = b.g_i(1, matrix_x, matrix_mu2, matrix_sigma2, prob_c=prob_c2)
g = g1 - g2

print('g1 =', g1)
print('g2 =', g2)
print('g1 - g2 =', g)
g = solve(g, x2)
lamb_g = lambdify(x1, g, 'numpy')

x_val = np.arange(-10, 10, 0.1)
y_val = lamb_g(x_val)
plt.plot(x_val, y_val[0])
plt.contour(x, y, z1)
plt.contour(x, y, z2)
plt.show()
