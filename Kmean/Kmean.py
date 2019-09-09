import numpy as np


class Kmean:
    def __init__(self, xy_arr, k):
        """
            k: number of centroid
            xy: list of (x, y) coordinate
        """
        self.xy_arr = xy_arr
        self.centroids = []
        self.k = k
        self.clusters = []
        self.random_centroids()

    def random_centroids(self):
        """
            random initial centriod
        """
        # initial centroid limit
        lim_max = np.max(self.xy_arr, axis=0)
        lim_min = np.min(self.xy_arr, axis=0)

        centroids = []
        for c in range(self.k):
            centroid = []
            for i in range(len(lim_max)):
                centroid.append(np.random.uniform(lim_min[i], lim_max[i]))
            centroids.append(centroid)
        self.centroids = np.array(centroids)

    def find_nearest_centroid(self, xy, centroids):
        """
            find nearest centriod from xy
            return nearest centriod index
        """
        square_dist = (np.array(xy) - np.array(centroids))**2
        sum_square_dist = np.sum(square_dist, axis=1)
        dist = np.sqrt(sum_square_dist)
        # find min centroid index
        min_ind = np.where(dist == np.min(dist))[0][0]
        return min_ind

    def clustering(self):
        """
            devide xy coordinate to k cluster
        """

        clusters = [[] for i in range(self.k)]
        for xy in self.xy_arr:
            min_ind = self.find_nearest_centroid(xy, self.centroids)
            clusters[min_ind].append(xy)
        self.clusters = np.array([np.array(c) for c in clusters])

    def update_centriod(self):
        """
            update old centroid:
            move centriod to center of cluster
        """
        new_centroids = []
        for i in range(self.k):
            try:
                new_centroid = np.mean(self.clusters[i], axis=0)
                new_centroids.append(new_centroid)
            except:
                new_centroids.append(self.centroids[i])
        self.centroids = np.array(new_centroids)

    def doKmean(self, max_iteration=1000):
        """
            start Kmean clustering for n iteration
        """
        # save old centroid
        old_centroid = self.centroids
        for i in range(max_iteration):
            print(i)
            self.clustering()
            self.update_centriod()

            # if new centroid is stable (nothing change from old centriod) stop kmean
            if np.equal(old_centroid, self.centroids).all():
                break

            old_centroid = self.centroids

    def doKmean_one_iter(self):
        self.clustering()
        self.update_centriod()


if __name__ == "__main__":
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    sns.set_context('notebook', font_scale=1.25, rc={"lines.linewidth": 3})

    ds = pd.read_csv('iris.csv')
    ds = ds.iloc[:, 3:5]
    xy = np.array(ds)

    k = 3
    kmn = Kmean(xy, k)
    kmn.doKmean(1000)

    fig = plt.figure(figsize=(9, 5), dpi=100)
    for i in range(k):
        sns.scatterplot(kmn.clusters[i][:, 0], kmn.clusters[i][:, 1], s=100)
        sns.scatterplot([kmn.centroids[i][0]],[kmn.centroids[i][1]], marker="X", s=200, color="k", alpha=0.75)
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.show()
