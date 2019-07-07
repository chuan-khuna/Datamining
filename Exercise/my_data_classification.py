import sys
sys.path.append("..")
import pandas as pd
import numpy as np
from LinearClassification import LinearClassification as LC
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("my_data.csv")

x = np.array(df[['x1', 'x2']])
aug_x = np.insert(x, 0, values=1, axis=1)
x1 = x[:, 0][0:100]
x2 = x[:, 1][0:100]

y = np.array(df['class'])[0:100]

plt.plot(x1[:50], x2[:50], 'b.')
plt.plot(x1[50:100], x2[50:100], 'g.')
plt.plot(x1[100:], x2[100:], 'r.')
plt.show()


boundaries = []
for i in range(2):
    # output for classify class i
    yi = np.where(y==i, 1, 0) 
    print(yi)
    init_w = np.ones(aug_x.shape[1])
    learning_rate = 0.2
    max_iteration = 2000
    lc = LC(aug_x, yi, init_w, learning_rate, max_iteration)
    boundaries.append(lc.parameters[-1])
    print(f'boundary for classify class {i}: {lc.parameters[-1]}')


for i in range(len(boundaries)):
    w0, w1, w2 = boundaries[i][0], boundaries[i][1], boundaries[i][2]
    x_val = np.arange(-5, 10, 0.1)
    y_val = -(-w1*x_val + w0)/w2
    plt.plot(x_val, y_val, label=f'b{i+1}, {w1:.2f}x1 + {w2:.2f}x2 + {w0:.2f} = 0')

plt.plot(x1[:50], x2[:50], 'bo', label='c1')
plt.plot(x1[50:100], x2[50:100], 'go', label='c2')
plt.plot(x1[100:], x2[100:], 'yo', label='c3')
plt.legend()
plt.grid(True, alpha=0.5)
plt.show()