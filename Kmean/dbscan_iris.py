from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

sns.set_style("whitegrid")
colors = sns.color_palette("muted")
sns.set_palette(colors)
sns.set_context("paper", font_scale=1.25, rc={"lines.linewidth": 1.5})

ds = pd.read_csv("iris.csv").iloc[:, 1:]

X = ds.iloc[:, 2:4]
species = ds["Species"]

clustering = DBSCAN(eps=0.15, min_samples=10)
clustering.fit(X)

predicted_label = clustering.labels_

fig = plt.figure(figsize=(8, 5), dpi=100)
plt.title("DBSCAN")
sns.scatterplot(X['Petal.Length'], X['Petal.Width'], hue=species, s=100, marker='s')
sns.scatterplot(X['Petal.Length'], X['Petal.Width'], hue=predicted_label)
plt.show()