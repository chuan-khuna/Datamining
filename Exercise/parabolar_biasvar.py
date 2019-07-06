import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import random
from BiasVar import BiasVar

n = 1000
x = np.linspace(-1, 1, n)
fx = x**2
x_aug = np.array([[1, i] for i in x])

num_model = 100
const_weights = np.random.uniform(0, 1, num_model)
const_weights_aug = np.array([[w, 0] for w in const_weights])

const_model = BiasVar(const_weights_aug, x_aug, fx)

print("------ constant model ------")
print("constant gbar: {}".format(const_model.mean_weight))
print("constant bias^2: {}".format(const_model.bias))
print("constant variance: {}".format(const_model.variance))
print("constant bias^2+variance: {}".format(const_model.bias+const_model.variance))

# first and second point index for linear model
fi = random.sample(range(n), num_model)
si = random.sample(range(n), num_model)

# create linear models
lin_weights = []
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
    lin_weights.append(model_w)
lin_weights = np.array(lin_weights)
lin_model = BiasVar(lin_weights, x_aug, fx)
print("------ linear model ------")
print("linear gbar: {}".format(lin_model.mean_weight))
print("linear bias^2: {}".format(lin_model.bias))
print("linear variance: {}".format(lin_model.variance))
print("linear bias^2+variance: {}".format(lin_model.bias+lin_model.variance))