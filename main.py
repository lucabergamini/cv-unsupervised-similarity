from __future__ import print_function, division
import numpy
import cv2
from utils import *
from matplotlib import pyplot


def histogram(image, mask):
    # extract a 3D color histogram from the masked region of the
    # image, using the supplied number of bins per channel; then
    # normalize the histogram
    hist = cv2.calcHist([image], [0, 1, 2], mask, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    # return the histogram
    return hist


img1 = cv2.imread("data/Animali/cane_1.JPG")
img2 = cv2.imread("data/Animali/cane_2.JPG")
print(numpy.sum(histogram(img1, None)))

maskes = get_mask(img1)
for mask in maskes:
    get_color_hist(img1, 128, mode="hsv", normalize_channel=True, mask=mask)


h1 = get_bgr_hist(img1, 256, True)
h2 = get_bgr_hist(img2, 256, True)

print(emd_from_hist(h1, h2,get_emd_matrix(256)))
s_1 = get_sift(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 50)
s_2 = get_sift(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), 50)
print(sift_match(s_1, s_2))

voc = get_BOW_vocabulary([s_1, s_2], cluster_number=10)
print(voc)
hist = get_BOW_hist(s_1, voc)
print(hist)
