import bayes_multi_attr as b
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from sklearn.naive_bayes import GaussianNB
from scipy.stats import norm
from sympy import *

df = pd.read_csv('iris2class.csv')
versicolor = df[0:50]
verginica = df[50:]

plt.title('Iris Data')
sns.scatterplot(x='Petal.Length', y='Petal.Width', data=df, hue='Species')
plt.show()

# sns.distplot(df['Petal.Length'][:50], kde=False, fit=norm, label='Versicolor')
# sns.distplot(df['Petal.Length'][50:], kde=False, fit=norm, label='Virginica')
# plt.legend()
# plt.show()

# sns.distplot(df['Petal.Width'][:50], kde=False, fit=norm, label='Versicolor')
# sns.distplot(df['Petal.Width'][50:], kde=False, fit=norm, label='Virginica')
# plt.legend()
# plt.show()

mu1 = np.array(versicolor.mean())
mu2 = np.array(verginica.mean())
sigma1 = np.array(versicolor.cov())
sigma2 = np.array(verginica.cov())

# versicolor
print('------ Versicolor ------')
print(f'mu: {mu1}')
print(f'sigma: {sigma1}')

# versicolor
print('------ Verginica ------')
print(f'mu: {mu2}')
print(f'sigma: {sigma2}')


# Bayes classifier
x_, y_ = symbols('x y')
matrix_x = Matrix([x_, y_]).T
matrix_mu1 = Matrix(mu1).T
matrix_mu2 = Matrix(mu2).T
matrix_sigma1 = Matrix(sigma1)
matrix_sigma2 = Matrix(sigma2)
g1 = b.g_i(1, matrix_x, matrix_mu1, matrix_sigma1, prob_c=0.5)
g2 = b.g_i(1, matrix_x, matrix_mu2, matrix_sigma2, prob_c=0.5)
g = g1 - g2
print(f'g = {g}')

g = solve(g, y_)
lamb_g = lambdify(x_, g, 'numpy')
petal_length_val = np.arange(0, 15, 0.001)
petal_width_val = lamb_g(petal_length_val)

# plt.plot(petal_length_val, petal_width_val[0], 'black', alpha=0.75)
# plt.plot(petal_length_val, petal_width_val[1], 'black', alpha=0.75)
# sns.scatterplot(x='Petal.Length', y='Petal.Width', data=df, hue='Species')
# plt.show()


x, y = np.mgrid[0:10:0.05, 0:10:0.05]
xy = np.dstack([x, y])
z1 = sp.stats.multivariate_normal.pdf(xy, mean=mu1, cov=sigma1)
z2 = sp.stats.multivariate_normal.pdf(xy, mean=mu2, cov=sigma2)
plt.contour(x, y, z1, cmap='Blues_r', alpha=0.5)
plt.contour(x, y, z2, cmap='Reds_r', alpha=0.5)
plt.contourf(x, y, z1/(z1+z2), alpha=0.75, cmap='Blues')
sns.scatterplot(x='Petal.Length', y='Petal.Width', data=df, hue='Species')
plt.plot(petal_length_val, petal_width_val[0], 'black', alpha=0.5)
plt.plot(petal_length_val, petal_width_val[1], 'black', alpha=0.5)
plt.show()
