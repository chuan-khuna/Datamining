import numpy as np
import matplotlib.pyplot as plt
import random

def bias(g_bar, fx):
    """
        bias:
            Ex[g_bar - fx]

        g_bar = mean of gD(x)       ; gD = model g from sample D
        fx = actual output
    """
    bias = np.mean((g_bar - fx)**2)
    return bias

def variance(g_d, g_bar):
    """
        variance = sum( (g_d - g_bar)**2 )/d_sample
    """
    variance = np.mean(
        (g_d - np.repeat(g_bar, repeats=len(g_d), axis=0))**2
    )
    return variance

if __name__ == "__main__":

    sin_sample = 100
    x = np.linspace(0, 2*np.pi, sin_sample)
    fx = np.sin(x)

    # number of gd model
    num_model = 10
    print("random {} samples of model".format(num_model))

    const_gd = np.random.uniform(-1, 1, num_model)
    const_gbar = np.mean(const_gd)
    const_bias = bias(np.repeat(const_gbar, sin_sample), fx)
    const_variance = variance(const_gd, const_gbar)
    print("constant gbar: {:.4f}".format(const_gbar))
    print("constant bias^2: {:.4f}".format(const_bias))
    print("constant variance: {:.4f}".format(const_variance))
    print("constant bias^2+variance: {:.4f}".format(const_bias+const_variance))