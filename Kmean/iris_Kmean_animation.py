import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from Kmean import Kmean


sns.set_style("whitegrid")
sns.set_palette("muted")
sns.set_context('notebook', font_scale=1.25, rc={"lines.linewidth": 3})

df = pd.read_csv("iris.csv")
df = df.iloc[:, 3:5]

xy = np.array(df)

max_iteration = 10
k = 3
kmn = Kmean(xy, 3)

fig = plt.figure(figsize=(8, 5), dpi=100)
plt.ion()
for i in range(max_iteration):
    plt.clf()
    centroids = kmn.centroids
    kmn.doKmean_one_iter()
    clusters = kmn.clusters
    plt.title(f"iteration {i+1}")
    for c in range(k):
        try:
            p = sns.scatterplot(clusters[c][:, 0], clusters[c][:, 1], s=100)
            current_palette = sns.color_palette()
            color = current_palette[c]
            sns.scatterplot([centroids[c][0]], [centroids[c][1]], marker="X", s=200, color=color, alpha=0.75, edgecolor='k')
        except:
            sns.scatterplot([centroids[c][0]], [centroids[c][1]], marker="X", s=200, color="black", alpha=0.75)
            
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.draw()
    plt.pause(0.5)
plt.ioff()
plt.show()
