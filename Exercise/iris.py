import sys
sys.path.append("..")
import pandas as pd
import numpy as np
from LinearClassification import LinearClassification as LC
import matplotlib.pyplot as plt

df = pd.read_csv("iris2class.csv")
print(df.head())

iris_class = np.array(df['Species'])
# encoding class to integer
# class 0 => setosa, class 1 => versicolor
iris_class = np.where(iris_class == "setosa", iris_class, 1)
iris_class = np.where(iris_class == "setosa", 0, iris_class)
iris_class.astype(np.double)

sl = df['Sepal.Length']
sw = df['Sepal.Width']
pl = df['Petal.Length']
pw = df['Petal.Width']
plt.plot(pl[:50], pw[:50], 'r.', label="Setosa")
plt.plot(pl[50:], pw[50:], 'b.', label="Versicolor")
plt.xlabel("Petal.Length")
plt.ylabel("Petal.Width")
plt.legend()
plt.show()

x = np.array(df[['Petal.Length', 'Petal.Width']])
x = np.insert(x, 0, values=1, axis=1)
learning_rate = 0.5
max_iteration = 100
# initaial weights
w = np.ones(x.shape[1])
lc = LC(x, iris_class, w, learning_rate, max_iteration)

params = lc.parameters[-1]
w0, w1, w2 = params[0], params[1], params[2]
axis_x = np.arange(0, 5, 0.1)
axis_y = -(w1*axis_x + w0)/w2
plt.plot(axis_x, axis_y, 'b-', label=f"boundary = {w1}x1 + {w2}x2 + {w0} = 0")
plt.plot(pl[:50], pw[:50], 'r.', label="Setosa")
plt.plot(pl[50:], pw[50:], 'b.', label="Versicolor")
plt.xlabel("Petal.Length")
plt.ylabel("Petal.Width")
plt.legend()
plt.show()