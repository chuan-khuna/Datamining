from Kmean import Kmean
from PIL import Image
import numpy as np
from copy import deepcopy
import time

def find_new_color(old_px, centroid_colors):
    # find nearest color
    square_dist = (np.array(old_px) - np.array(centroid_colors))**2
    sum_square_dist = np.sum(square_dist, axis=1)
    dist = np.sqrt(sum_square_dist)
    min_ind = np.where(dist == np.min(dist))[0][0]
    new_color = centroid_colors[min_ind]
    return new_color


def KmeanImgCompress(file_name, kmn_centroid=4, kmn_iter=10):
    img = Image.open(file_name)
    img_w, img_h = img.size
    img_px = np.array(img.getdata())
    new_img_px = deepcopy(img_px)

    kmn = Kmean(img_px, kmn_centroid)
    kmn.doKmean(kmn_iter)

    centroid_colors = np.round(kmn.centroids)

    for px in range(len(img_px)):
        new_color = find_new_color(img_px[px], centroid_colors)
        new_img_px[px] = new_color

    # convert to tuple
    new_img_px = [tuple(px) for px in new_img_px]
    output_img = Image.new("RGB", (img_w, img_h))
    output_img.putdata(new_img_px)

    output_img.save(f'{file_name[:-4]}_Quantuzed.jpg')

if __name__ == "__main__":
    file_name = "sniper.jpg"
    color = 16
    max_iteration = 10
    start_time = time.time()
    kmn_img = KmeanImgCompress(file_name, color, max_iteration)
    elapsed = np.round(time.time() - start_time, 3)
    print(f"compress complete, elapsed time: {elapsed}\ts")