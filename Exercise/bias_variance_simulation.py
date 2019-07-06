import numpy as np
import matplotlib.pyplot as plt
import random

sin_sample = 1000
x = np.linspace(0, 2*np.pi, sin_sample)
y = np.sin(x)

sample = 100
rand_const = np.random.uniform(-1, 1, sample)

print("------ constant model ------")
print("random {} samples constant".format(sample))

for c in rand_const:
    plt.plot(x, [c]*len(x), alpha=0.2)

# g bar
const_g_bar = round(np.mean(rand_const), 5)
print("random constant mean: {}".format(const_g_bar))

# bias = g bar - fx
const_bias = np.array([const_g_bar] * len(y)) - y
const_bias = np.sum(const_bias)
print("const bias = {}".format(const_bias))

# variance
cosnt_variance = np.mean(
    (rand_const - const_g_bar)**2
)
print("const variance = {}".format(cosnt_variance))

print("------ linear model ------")


# first and second point index
fi = random.sample(range(sin_sample), sample)
si = random.sample(range(sin_sample), sample)

models_w = []
for i in range(len(fi)):
    x1 = x[fi[i]]
    y1 = y[fi[i]]
    while fi[i] == si[i]:
        # random agian if same index
        si[i] = random.randint(0, len(fi)-1)
    x2 = x[si[i]]
    y2 = y[si[i]]
    
    m = (y1 - y2)/(x1 - x2)
    c = y1 - m*x1
    model_w = [c, m]
    models_w.append(model_w)

models_w = np.array(models_w)
for i in range(len(models_w)):
    plt.plot(x, x*models_w[i][1]+models_w[i][0], alpha=0.2)
    pass

# g bar, y_bar = m_bar*x + c_bar
m_bar = np.mean(models_w[:, 1])
c_bar = np.mean(models_w[:, 1])
print("random linear mean: {}x + {}".format(round(m_bar, 3), round(c_bar, 3)))
# linear g bar
l_g_bar = m_bar*x + c_bar
# bias = g bar - fx
l_bias = np.sum(l_g_bar - y)
print("linear bias = {}".format(l_bias))


plt.plot(x, [const_g_bar]*len(x), 'r-', label="mean of const random")
plt.plot(x, m_bar*x + c_bar, 'g-', label="mean of linear random")
plt.plot(x, y, 'b-')
plt.axis([0, 2*np.pi, -1.25, 1.25])
plt.grid(True, alpha=0.8)
plt.legend()
plt.show()
