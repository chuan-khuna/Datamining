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


def g_i(num_attr, x, mu, sigma, prob_c, prob_x):
    # Quadratic Discriminant Gi
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
    first = (mu1 - mu2).T.dot(sigma.inv().dot(x))
    second = -0.5*(
        (mu1 - mu2).T.dot(
            (sigma.inv().dot(mu1 + mu2))
        )
    )
    third = ln(prob_c1/prob_c2)
    return first + second + third
