from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

sns.set_style("whitegrid")
colors = sns.color_palette("muted")
sns.set_palette(colors)
sns.set_context("paper", font_scale=1.25, rc={"lines.linewidth": 1.5})

ds = pd.read_csv("iris.csv").iloc[:, 1:]

X = ds.iloc[:,:-1]
species = ds["Species"]

methods = ["single", "average", "complete", "ward"]
method = 1

linked = linkage(X, method=methods[method])
fig = plt.figure(figsize=(8, 5), dpi=100)
plt.title("scipy linkage")
dendrogram = dendrogram(linked, orientation='left', distance_sort=False)
plt.grid(False)
plt.show()


cluster = AgglomerativeClustering(n_clusters=3, linkage=methods[method])
cluster.fit_predict(X)
print(cluster.labels_)
fig = plt.figure(figsize=(8, 5), dpi=100)
plt.title("sklearn AgglomerativeClustering")
sns.scatterplot(X['Petal.Length'], X['Petal.Width'], hue=ds["Species"], cmap=colors, marker="s", s=200)
sns.scatterplot(X['Petal.Length'], X['Petal.Width'], hue=cluster.labels_)
plt.show()

# fig = plt.figure(figsize=(8, 5), dpi=100)
row_colors = species.map(dict(zip(species.unique(), colors[:len(species.unique())])))
sns.clustermap(X, method=methods[method], cmap="mako", row_colors=row_colors)
plt.title("seaborn clustermap")
plt.show()