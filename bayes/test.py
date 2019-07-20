import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd
import scipy as sp
sns.set_style("whitegrid")
sns.set_palette("bright")
sns.set_context('notebook', font_scale=1.25, rc={"lines.linewidth": 3})

c_map = 'rainbow'

x, y = np.mgrid[-10:10:0.1, -10:10:0.1]

# stack to (x, y) pair
xy = np.dstack([x, y])

mu = [-2, -1]
sigma = [[1, 0], [0, 2]]

z = sp.stats.multivariate_normal.pdf(xy, mean=mu, cov=sigma)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z,
                rstride=5, cstride=5,
                cmap=c_map, edgecolor='none')
plt.contour(x, y, z, zdir='z', offset=-0.05, cmap=c_map)
mu = [4, 4]
sigma = [[2, 0], [0, 4]]
z = sp.stats.multivariate_normal.pdf(xy, mean=mu, cov=sigma)
ax.plot_surface(x, y, z,
                rstride=5, cstride=5,
                cmap=c_map, edgecolor='none')
plt.contour(x, y, z, zdir='z', offset=-0.05, cmap=c_map)
plt.show()

plt.contour(x, y, z, cmap=c_map)
plt.contour(x, y, z, cmap=c_map)
plt.axis([-10, 10, -10, 10])
plt.show()
