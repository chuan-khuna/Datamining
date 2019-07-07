import sys
sys.path.append("..")
import pandas as pd
import numpy as np
from LinearClassification import LinearClassification as LC
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

df = pd.read_csv("iris.csv")
x = np.array(df[['Sepal.Length', 'Sepal.Width']])
aug_x = np.insert(x, 0, values=1, axis=1)
x1 = x[:, 0]
x2 = x[:, 1]
y = np.array(df['Species'])

species = ["setosa", "versicolor", "virginica"]

boundaries = []
for i in range(3):
    # output for classify class i
    yi = np.where(y==species[i], 1, 0) 
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
    sns.lineplot(x_val, y_val, label=f'b{i+1}, {w1:.2f}x1 + {w2:.2f}x2 + {w0:.2f} = 0', alpha=0.8)


# Kesler consrtuction
# w12 = boundaries[0] - boundaries[1]
# w23 = boundaries[1] - boundaries[2]
# w13 = boundaries[0] - boundaries[2]
# sns.lineplot(np.arange(0, 6.4, 0.1), -(w12[1]*np.arange(0, 6.4, 0.1) + w12[0])/w12[2], label=f'k1-2')
# sns.lineplot(np.arange(6.3, 15, 0.1), -(w23[1]*np.arange(6.3, 15, 0.1) + w23[0])/w23[2], label=f'k2-3')
# sns.lineplot(np.arange(6.3, 8, 0.1), -(w13[1]*np.arange(6.3, 8, 0.1) + w13[0])/w13[2], label=f'k1-3')


sns.scatterplot(x1[:50], x2[:50], label='c1')
sns.scatterplot(x1[50:100], x2[50:100], label='c2')
sns.scatterplot(x1[100:], x2[100:], label='c3')
plt.axis(
    [np.amin(x1)-1, np.amax(x1)+1,np.amin(x2)-1, np.amax(x2)+1]
)
plt.legend()
plt.show()