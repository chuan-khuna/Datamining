import matplotlib.pyplot as plt
import numpy as np

mean = [3, 0]
cov = [[1, 0], [0, 1]]  # diagonal covariance

x1, y1 = np.random.multivariate_normal(mean, cov, 25).T

mean = [6.5, 6]
cov = [[0.3, 0], [0, 0.3]]  # diagonal covariance

x2, y2 = np.random.multivariate_normal(mean, cov, 25).T

mean = [12, 1]
cov = [[0.5, 0], [0, 1.3]]  # diagonal covariance

x3, y3 = np.random.multivariate_normal(mean, cov, 25).T

plt.plot(x1, y1, 'r.')
plt.plot(x2, y2, 'g.')
plt.plot(x3, y3, 'b.')

x = np.concatenate([
    x1, x2, x3
], axis=0)

y = np.concatenate([
    y1, y2, y3
], axis=0)

out_class = np.concatenate([
    np.zeros(25), np.ones(25), np.ones(25)*2
], axis=0)

my_data = {
    'x1': x, 'x2': y, 'class': out_class
}

df = pd.DataFrame(my_data, columns= ['x1', 'x2', 'class'])
df.to_csv('./my_data.csv', index = None, header=True)
plot.show()