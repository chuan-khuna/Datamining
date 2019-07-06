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
        (g_d - g_bar)**2
    )
    return variance

if __name__ == "__main__":

    sin_sample = 10000
    x = np.linspace(0, 2*np.pi, sin_sample)
    fx = np.sin(x)

    # number of gd model
    num_model = 1000
    print("random {} samples of model".format(num_model))

    const_gd = np.random.uniform(-1, 1, num_model)
    const_gbar = np.mean(const_gd)
    const_bias = bias(np.repeat(const_gbar, sin_sample), fx)
    const_variance = variance(const_gd, np.repeat(const_gbar, repeats=num_model))
    print("------ constant model ------")
    print("constant gbar: {:.4f}".format(const_gbar))
    print("constant bias^2: {:.4f}".format(const_bias))
    print("constant variance: {:.4f}".format(const_variance))
    print("constant bias^2+variance: {:.4f}".format(const_bias+const_variance))

    # first and second point index for linear model
    fi = random.sample(range(sin_sample), num_model)
    si = random.sample(range(sin_sample), num_model)
    lin_gd = []
    # create linear models, 
    for i in range(num_model):
        x1 = x[fi[i]]
        y1 = fx[fi[i]]
        while fi[i] == si[i]:
            # random agian if same index
            si[i] = random.randint(0, len(fi)-1)
        x2 = x[si[i]]
        y2 = fx[si[i]]
        
        m = (y1 - y2)/(x1 - x2)
        c = y1 - m*x1
        model_w = [c, m]
        lin_gd.append(model_w)
    lin_gd = np.round(np.array(lin_gd), 5)
    lin_gbar = np.array([
        np.mean(lin_gd[:, 0]), np.mean(lin_gd[:, 1])
    ])
    lin_bias = bias(lin_gbar[1]*x + lin_gbar[0], fx)
    lin_variance = variance(
        np.array([x*w1 + w0 for (w0, w1) in lin_gd]),
        np.repeat(np.array([lin_gbar[0]+lin_gbar[1]*x]), repeats=num_model, axis=0)
    )
    print("------ linear model ------")
    print("linear gbar: {}".format(lin_gbar))
    print("linear bias^2: {:.4f}".format(lin_bias))
    print("linear variance: {:.4f}".format(lin_variance))
    print("linear bias^2+variance: {:.4f}".format(lin_bias+lin_variance))