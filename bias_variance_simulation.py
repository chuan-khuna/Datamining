import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 2*np.pi, 0.001)
y = np.sin(x)

sample = 10
rand_const = np.random.uniform(-1, 1, sample)

print("------ constant model ------")
print("random {} samples constant".format(sample))

# for c in rand_const:
#     plt.plot(x, [c]*len(x), alpha=0.5)

# g bar
rand_mean = round(np.mean(rand_const), 5)
print("random constant mean: {}".format(rand_mean))

# bias = g bar - fx
bias = np.array([rand_mean] * len(y)) - y
bias = np.sum(bias)
print("bias = {}".format(bias))

# variance
variance = np.mean(
    (rand_const - rand_mean)**2
)
print("variance = {}".format(variance))

print("------ linear model ------")

params = []
for i in range(sample):
    fi = np.random.randint(0, len(x)-1)
    si = np.random.randint(0, len(x)-1)
    wh


plt.plot(x, [rand_mean]*len(x), 'r-', label="mean of uniform random")
plt.plot(x, y)
plt.grid(True, alpha=0.8)
plt.show()
