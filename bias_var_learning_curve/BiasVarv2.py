import numpy as np
import matplotlib.pyplot as plt
import random


def cal_bias(gavg_x, y):
    bias = np.mean((gavg_x - y)**2)
    return np.round(bias, 5)


def cal_variance(model_results, gavg_x):
    sse = (np.repeat([gavg_x], len(model_results), axis=0) - model_results)**2
    variance = np.mean(sse)
    return np.round(variance, 5)

def cal_gavg(model_weights):
    mean_w = np.mean(model_weights, axis=0)
    return np.round(mean_w, 5)

def cal_e_in(g_xi, f_xi):
    in_sample_err = np.mean((g_xi - f_xi)**2)
    return np.round(in_sample_err, 5)

def cal_e_out(gx, fx):
    out_sample_err = np.mean((gx - fx)**2)
    return np.round(out_sample_err, 5)

if __name__ == '__main__':
    n = 200
    x = np.linspace(-1, 1, n)
    fx = np.sin(np.pi*x)
    mean_w = 0
    mean_w_res = np.array([mean_w]*200)
    print(cal_bias(mean_w_res, fx))
