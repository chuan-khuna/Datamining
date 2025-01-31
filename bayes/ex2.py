import bayes_single_attr as b
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
sns.set_style("whitegrid")
sns.set_palette("bright")
sns.set_context('notebook', font_scale=1.25, rc={"lines.linewidth": 3})

mu1 = -1
mu2 = 2
sigma1 = 2
sigma2 = 3
num_sample = 200
boundary = b.double_threshold_boundary(mu1, mu2, sigma1, sigma2)
x = np.linspace(-15, 15, num_sample)
x_lim = [-15, 15]
y1 = sp.stats.norm.pdf(x, mu1, sigma1)
y2 = sp.stats.norm.pdf(x, mu2, sigma2)
# b.likelihood_plot(x, y1, y2, boundary, x_lim, [-0.01, 0.3])
# b.posterior_plot(x, y1, y2, boundary, x_lim)


rand_size = 200
x1 = np.random.normal(loc=mu1, scale=sigma1, size=rand_size)
x2 = np.random.normal(loc=mu2, scale=sigma2, size=rand_size)
sns.distplot(x1)
sns.distplot(x2)
mu1 = np.mean(x1)
mu2 = np.mean(x2)
sigma1 = np.sqrt(np.sum((x1 - mu1)**2)/len(x1))
sigma2 = np.sqrt(np.sum((x2 - mu2)**2)/len(x2))
boundary = b.double_threshold_boundary(mu1, mu2, sigma1, sigma2)
print(f'mu1: {mu1:.2f}, sigma1: {sigma1:.2f}')
print(f'mu2: {mu2:.2f}, sigma2: {sigma2:.2f}')
y1 = sp.stats.norm.pdf(x, mu1, sigma1)
y2 = sp.stats.norm.pdf(x, mu2, sigma2)
b.likelihood_plot(x, y1, y2, boundary, x_lim, [-0.01, 0.3])
b.posterior_plot(x, y1, y2, boundary, x_lim)
