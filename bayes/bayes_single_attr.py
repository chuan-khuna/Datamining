import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import math

sns.set_style("whitegrid")
sns.set_palette("bright")
sns.set_context('notebook', font_scale=1.25, rc={"lines.linewidth": 3})


def likelihood_plot(x, y1, y2, boundary, x_lim=[-5, 5], y_lim=[-0.02, 0.5]):
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.lineplot(x, y1, label='c1')
    sns.lineplot(x, y2, label='c2')
    for b in boundary:
        plt.axvline(x=b, linewidth=2, color='black',
                    alpha=0.75, label=f'boundary: x={b}')
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
    for b in boundary:
        plt.axvline(x=b, linewidth=2, color='black',
                    alpha=0.75, label=f'boundary: x={b}')
    plt.ylabel('posterior prob.')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    plt.legend()
    plt.show()


def single_threshold_boundary(mu1, mu2):
    """
        boundary for bayes classification
        - single attribute
        - sigma1 = sigma2
        - prior prob = gaussian (normal distribution) 
    """
    return [round((mu1 + mu2)/2, 3)]


def double_threshold_boundary(mu1, mu2, sigma1, sigma2):
    """
        boundary for bayes classification
        - single attribute
        - sigma1 != sigma2
        - prior prob = gaussian (normal distribution)

        g(x) = -(x-(m))^2/(2(s)^2) - ln(s) + (x-(n))^2/(2(t)^2) + ln(t) = 0 
        # where m = mu1, n = mu2, s = sigma1, t = sigma 2
        # wolfram: 
        # https://www.wolframalpha.com/input/?i=-(x-(m))%5E2%2F(2(s)%5E2)+-+ln(s)+%2B+(x-(n))%5E2%2F(2(t)%5E2)+%2B+ln(t)+%3D+0
    """
    num = ((mu2/sigma2**2 - mu1/sigma1**2))
    root = math.sqrt((mu1/sigma1**2 - mu2/sigma2**2)**2 - 4*(1/(2*sigma2**2) - 1/(2*(sigma1**2)))
                     * (-mu1**2/(2*(sigma1**2)) + mu2**2/(2*(sigma2**2)) - np.log(sigma1) + np.log(sigma2)))
    denominator = (2*(1/(2*(sigma2**2)) - 1/(2*(sigma1**2))))
    boundary1 = round((num+root)/denominator, 3)
    boundary2 = round((num-root)/denominator, 3)
    return [boundary1, boundary2]
