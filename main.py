from __future__ import print_function, division
import numpy
import cv2
from utils import *
from matplotlib import pyplot

img1 = cv2.imread("data/cat_1.jpg")
img2 = cv2.imread("data/cat_2.jpg")
h1 = get_bgr_hist(img1, 256, True)
h2 = get_bgr_hist(img2, 256, True)

print(emd_from_hist(h1,h2))

s_1 = get_sift(cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY),50)
s_2 = get_sift(cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY),50)
print(sift_match(s_1,s_2))


