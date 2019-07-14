import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

sns.set_style("whitegrid")
sns.set_palette("bright")
sns.set_context('notebook', font_scale=1.25, rc={"lines.linewidth": 3})

def likelihood_plot(x, y1, y2, boundary):
    x_lim = [-5, 5]
    y_lim = [-0.02, 0.5]
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.lineplot(x, y1, label='c1')
    sns.lineplot(x, y2, label='c2')
    plt.ylabel('likelihood')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    plt.legend()
    plt.show()

def posterior_plot(x, y1, y2, boundary):
    x_lim = [-10, 10]
    y_lim = [-0.1, 1.1]
    fig, ax = plt.subplots(figsize=(9, 6))
    post1 = y1 / (y1+y2)
    post2 = 1 - post1
    sns.lineplot(x, post1, label='c1')
    sns.lineplot(x, post2, label='c2')
    plt.ylabel('posterior prob.')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    plt.legend()
    plt.show()

mu1 = -1
mu2 = 1
sigma1 = 1
sigma2 = 2
num_sample = 200
boundary = (mu1 + mu2)/2
x = np.linspace(-10, 10, num_sample)
y1 = sp.stats.norm.pdf(x, mu1, sigma1)
y2 = sp.stats.norm.pdf(x, mu2, sigma2)
likelihood_plot(x, y1, y2, boundary)
posterior_plot(x, y1, y2, boundary)