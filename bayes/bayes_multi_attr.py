import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from sympy import *
from mpmath import *
init_printing(use_unicode=True)

sns.set_style("whitegrid")
sns.set_palette("bright")
sns.set_context('notebook', font_scale=1.25, rc={"lines.linewidth": 3})


def g_i(num_attr, x, mu, sigma, prob_c, prob_x=0):
    # Quadratic Discriminant gi
    first = -0.5 * (x.T).dot((sigma.inv()).dot(x))
    second = (mu.T).dot((sigma.inv()).dot(x))
    third = -0.5 * (mu.T).dot((sigma.inv()).dot(mu))
    fourth = -(num_attr/2)*ln(2*pi)
    fifth = -0.5*ln(sigma.det())
    sixth = ln(prob_c)
    seventh = -ln(prob_x)
    seventh = 0

    return first + second + third + fourth + fifth + sixth + seventh


def g_lin(mu1, mu2, sigma, x, prob_c1, prob_c2, prob_x=0):
    # linear discriminant gij
    first = (mu1 - mu2).T.dot(sigma.inv().dot(x))
    second = -0.5*(
        (mu1 - mu2).T.dot(
            (sigma.inv().dot(mu1 + mu2))
        )
    )
    third = ln(prob_c1/prob_c2)
    return first + second + third


def surface_plot(x, y, z_list):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    c_map = 'plasma'
    for z in z_list:
        ax.plot_surface(x, y, z,
                        rstride=5, cstride=5,
                        cmap=c_map, edgecolor='none', zorder=0.6)
        plt.contour(x, y, z, zdir='z', offset=-0.05, cmap=c_map)
    plt.show()


def contour_plot(x, y, z_list):
    for z in z_list:
        plt.contour(x, y, z)
    plt.show()
