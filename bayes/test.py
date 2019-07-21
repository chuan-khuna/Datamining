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

z1 = sp.stats.multivariate_normal.pdf(xy, mean=mu, cov=sigma)
fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x, y, z1,
#                 rstride=5, cstride=5,
#                 cmap=c_map, edgecolor='none')
# plt.contour(x, y, z1, zdir='z', offset=-0.05, cmap=c_map)
mu = [-2, 3]
sigma = [[2, 1], [1, 6]]
z2 = sp.stats.multivariate_normal.pdf(xy, mean=mu, cov=sigma)
# ax.plot_surface(x, y, z2,
#                 rstride=5, cstride=5,
#                 cmap=c_map, edgecolor='none')
# plt.contour(x, y, z2, zdir='z', offset=-0.05, cmap=c_map)
# plt.show()

posterior = z1/(z1+z2)

plt.contour(x, y, z1, cmap='Blues_r')
plt.contour(x, y, z2, cmap="Reds_r")
# plt.contourf(x, y, posterior, cmap="Blues", alpha=0.5)
# plt.contourf(x, y, z2/(z1+z2), cmap="Reds", alpha=0.5)
plt.axis([-10, 10, -10, 10])
plt.show()
