import sys
sys.path.append("..")
import pandas as pd
import numpy as np
from LinearClassification import LinearClassification 
import matplotlib.pyplot as plt
import time


# AI page 99

x = np.array([
    [1, 0, 0, -1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, -1, 0, 0],
    [-1, -1, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, -1, -1, 0],
    [-1, 0, 0, 0, 0, 0, 1, 0, 0],
    [-1, -1, -1, 0, 0, 0, 1, 1, 1]
])

w = np.array([
    0.5, -1, -1, -0.5, 1, -1, 1.5, 1, 1.5
])

y = np.ones(6)

lc = LinearClassification(x, y, w, 0.5, 1000)
p = lc.parameters[-1]
print(p)

axis_x = np.arange(-0.5, 1.5, 0.01)
b1 = -(p[1]*axis_x + p[0])/p[2]
b2 = -(p[4]*axis_x + p[3])/p[5]
b3 = -(p[7]*axis_x + p[6])/p[8]

plt.scatter(0, 0, label="c1")
plt.scatter(1, 0, label="c2")
plt.scatter(1, 1, label="c3")

plt.plot(axis_x, b1, label='c1 boundary', alpha=0.3)
plt.plot(axis_x, b2, label='c2 boundary', alpha=0.3)
plt.plot(axis_x, b3, label='c3 boundary', alpha=0.3)

k12 = b1 - b2
k23 = b2 - b3
k13 = b1 - b3
plt.plot(axis_x, k12, label='kesler1-2 boundary')
plt.plot(axis_x, k23, label='kesler2-3 boundary')
plt.plot(axis_x, k13, label='kesler1-3 boundary')


plt.grid(True, alpha=0.3)
plt.legend()
plt.show()