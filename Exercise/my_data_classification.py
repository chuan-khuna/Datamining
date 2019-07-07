import sys
sys.path.append("..")
import pandas as pd
import numpy as np
from LinearClassification import LinearClassification as LC
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

df = pd.read_csv("my_data.csv")
x = np.array(df[['x1', 'x2']])
aug_x = np.insert(x, 0, values=1, axis=1)
x1 = x[:, 0]
x2 = x[:, 1]
y = np.array(df['class'])

boundaries = []
for i in range(3):
    # output for classify class i
    yi = np.where(y==i, 1, 0) 
    init_w = np.ones(aug_x.shape[1])
    learning_rate = 0.2
    max_iteration = 500
    lc = LC(aug_x, yi, init_w, learning_rate, max_iteration)
    boundaries.append(lc.parameters[-1])
    print(f'boundary for classify class {i}: {lc.parameters[-1]}')

x_val = np.arange(np.amin(x1)-5, np.amax(x1)+5, 0.1)
for i in range(len(boundaries)):
    w0, w1, w2 = boundaries[i][0], boundaries[i][1], boundaries[i][2]  
    y_val = -(w1*x_val + w0)/w2
    sns.lineplot(x_val, y_val, label=f'b{i+1}, {w1:.2f}x1 + {w2:.2f}x2 + {w0:.2f} = 0')


sns.scatterplot(x1[:50], x2[:50], label='c1')
sns.scatterplot(x1[50:100], x2[50:100], label='c2')
sns.scatterplot(x1[100:], x2[100:], label='c3')
plt.axis(
    [np.amin(x1)-1, np.amax(x1)+1,np.amin(x2)-1, np.amax(x2)+1]
)
plt.legend()
plt.show()