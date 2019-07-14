import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

sns.set_style("whitegrid")
sns.set_palette("bright")
sns.set_context('notebook', font_scale=1.25, rc={"lines.linewidth": 3})

def likelihood_plot(x, y1, y2, boundary, x_lim=[-5, 5], y_lim=[-0.02, 0.5]):
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.lineplot(x, y1, label='c1')
    sns.lineplot(x, y2, label='c2')
    plt.axvline(x=boundary, linewidth=2, color='black', alpha=0.75, label=f'boundary: x={boundary}')
    plt.ylabel('likelihood')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    plt.legend()
    plt.show()

def posterior_plot(x, y1, y2, boundary, x_lim=[-5, 5], y_lim=[-0.1, 1.1]):
    fig, ax = plt.subplots(figsize=(9, 6))
    posterior_prob1 = y1/(y1+y2)
    posterior_prob2 = 1 - posterior_prob1
    sns.lineplot(x, posterior_prob1, label='c1')
    sns.lineplot(x, posterior_prob2, label='c2')
    plt.axvline(x=boundary, linewidth=2, color='black', alpha=0.75, label=f'boundary: x={boundary}')
    plt.ylabel('posterior prob.')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    plt.legend()
    plt.show()

def boundary(mu1, mu2):
    """
        boundary for bayes classification
        - single attribute
        - sigma1 = sigma2
        - prior prob = gaussian (normal distribution) 
    """
    return (mu1 + mu2)/2
