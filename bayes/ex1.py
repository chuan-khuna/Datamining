import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

sns.set_style("whitegrid")
sns.set_palette("muted")
sns.set_context('notebook', font_scale=1.25)

mu1 = -1
mu2 = 1
sigma1 = 1
sigma2 = 1
num_sample = 500


num_sample = 200
x = np.linspace(-10, 10, num_sample)

y1 = sp.stats.norm.pdf(x, mu1, sigma1)
y2 = sp.stats.norm.pdf(x, mu2, sigma2)


fig = plt.figure(figsize=(9, 6))
sns.lineplot(x, y1, label='c1')
sns.lineplot(x, y2, label='c2')
plt.ylabel('likelihood')
plt.axis([-6, 6, -0.02, 0.5])
plt.legend()
plt.savefig('ex1-likelihood.png', dpi=200, bbox_inches='tight')
plt.show()

