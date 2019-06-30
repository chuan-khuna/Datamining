import sys
sys.path.append("..")
import pandas as pd
import numpy as np
from LinearRegression import LinearRegression as LR
from VisualizationLR import Visualization as VLR
from LeastSquare import LeastSquare as LS
import matplotlib.pyplot as plt

num_sample = 100
m = 2
c = 5
min_x = -5
max_x = 10
x = np.linspace(min_x, max_x, num_sample)
y = m*x + c + np.random.randn(num_sample)
X = np.array([[1, i] for i in x])

print("Expected: y = {}x + {} + (random)".format(m, c))
ls = LS(X, y)
print("Least Square: h = {}x + {}".format(ls.parameters[1], ls.parameters[0]))


max_iteration = 1000
learning_rate = 0.0001
lr = LR(x, y, learning_rate=learning_rate, 
        max_iteration=max_iteration)
print("Linear Regression: h = {}x + {}".format(lr.parameters[-1][1], lr.parameters[-1][0]))
plt.title('Linear Equation')
plt.plot(x, y, 'r.')
plt.plot(x, lr.predicted_y, 'b-')
plt.show()

print("--Orange without standardization--")
df = pd.read_csv("orange_5.csv", sep=',')
age = np.array(df['age'])
cir = np.array(df['circumference'])
X = np.array([[1, i] for i in age])
ls = LS(X, cir)
print("Least Square: h = {}x + {}".format(ls.parameters[1], ls.parameters[0]))
x = np.linspace(np.amin(age), np.amax(age), 100)
predicted_y = ls.parameters[0] + ls.parameters[1]*x
plt.plot(age, cir, 'r.')
plt.plot(x, predicted_y, 'b-')
plt.title('Least Square Orange without Standardization')
plt.xlabel("age")
plt.ylabel("cir")
plt.show()

max_iteration = 3
learning_rate = 0.00001
lr = LR(age, cir, learning_rate=learning_rate, max_iteration=max_iteration)
print("Linear Regression: h = {}x + {}".format(lr.parameters[-1][1], lr.parameters[-1][0]))


print("--Orange with standardization--")
s_age = (age - np.mean(age))/np.std(age)
s_cir = (cir - np.mean(cir))/np.std(cir)
max_iteration = 100
learning_rate = 0.2
s_lr = LR(s_age, s_cir, learning_rate=learning_rate, max_iteration=max_iteration)
X = np.array([[1, i] for i in s_age])
s_ls = LS(X, s_cir)
print("Least Square: h = {}x + {}".format(s_ls.parameters[1], s_ls.parameters[0]))
print("Linear Regression: h = {}x + {}".format(s_lr.parameters[-1][1], s_lr.parameters[-1][0]))
x = np.linspace(np.amin(s_age), np.amax(s_age), 100)
predicted_y = s_ls.parameters[0] + s_ls.parameters[1]*x
plt.plot(s_age, s_cir, 'r.')
plt.plot(x, predicted_y, 'b-')
plt.show()