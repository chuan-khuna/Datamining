import numpy as np


class Otsu:
    def __init__(self):
        self.threshold = 0
        self.inter_class_var = []

    def fit(self, hist_counts, hist_val):
        self.counts = hist_counts
        self.hist_val = hist_val

        max_ind = 0
        max_var = 0
        for i in range(1, len(self.hist_val)):
            print(f"current theshold: {self.hist_val[i]}")
            c1_val = np.where(self.hist_val < self.hist_val[i])
            c2_val = np.where(self.hist_val >= self.hist_val[i])
            c1_hist = self.counts[:i]
            c2_hist = self.counts[i:]
            w1 = np.sum(c1_hist)
            w2 = np.sum(c2_hist)
            mu1 = np.sum(c1_val * c1_hist) / w1
            mu2 = np.sum(c2_val * c2_hist) / w2

            inter_class_var = (w1 * w2) * (mu1 - mu2) ** 2
            if np.isnan(inter_class_var):
                inter_class_var = -1
            self.inter_class_var.append(inter_class_var)
            if max_var < inter_class_var:
                max_var = inter_class_var
                max_ind = i

        self.threshold = self.hist_val[max_ind]
        self.inter_class_var = np.array(self.inter_class_var)


if __name__ == "__main__":

    from PIL import Image
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    sns.set_style("whitegrid")
    colors = sns.color_palette("muted")
    sns.set_palette(colors)
    sns.set_context("paper", font_scale=1.25, rc={"lines.linewidth": 3})

    img = Image.open("sniper.jpg")
    bw = img.convert("L")
    bw_px = np.array(bw.getdata())

    color_range = np.arange(0, 256)
    bw_px_hist = np.bincount(bw_px)

    otsu = Otsu()
    otsu.fit(bw_px_hist, color_range)

    theshold = otsu.threshold
    inter_var = otsu.inter_class_var

    otsu_img = Image.new("L", img.size)
    otsu_img.putdata(np.where(bw_px < otsu.threshold, 0, 255))
    otsu_img.save("Otsu.jpg")

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), dpi=150, sharex=True)
    sns.distplot(bw_px, bins=color_range, ax=axs[0])
    axs[0].set_ylim([0, 0.01])

    sns.lineplot(color_range[1:], inter_var, ax=axs[1])
    plt.axvline(theshold, color=colors[1], label=f"threshold: {theshold}")
    plt.legend()
    plt.show()

    fig, axs = plt.subplots(1, 3, dpi=150)
    axs[0].imshow(np.asarray(img))
    axs[1].imshow(np.asarray(bw), cmap="gray", vmin=0, vmax=255)
    axs[2].imshow(np.asarray(otsu_img), cmap="gray", vmin=0, vmax=255)
    for ax in axs:
        ax.grid(False)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
    plt.show()
