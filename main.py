from __future__ import print_function, division
import numpy
import cv2
from utils import get_bgr_hist,emd_from_hist,get_hog
from matplotlib import pyplot

img1 = cv2.imread("data/cat.jpg")
img2 = cv2.imread("data/cat_2.jpg")
h1 = get_bgr_hist(img1, 256, True)
h2 = get_bgr_hist(img2, 256, True)

print(emd_from_hist(h1,h2))



