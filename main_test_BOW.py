from __future__ import print_function, division
import numpy
import cv2
from utils import *
from matplotlib import pyplot

#carico data
data = numpy.load("bow_75_1500.npy").item()

#prendo tutti istogrammi per ricerche
hists_sift = numpy.asarray([i["hist_sift"] for i in data["img_data"]])
hists_colors = numpy.asarray([i["hist_color"] for i in data["img_data"]])
# matrice
matrix = get_emd_matrix(8)

for dictionary in data["img_data"]:
    # prendo hist sift
    hist_sift = dictionary["hist_sift"]
    # prendo hist colore
    hist_color = dictionary["hist_color"]
    #adesso devo cercare i piu simili
    # metrica combinata con sift e colori
    # calcolo distanza L2 sugli hist
    # quello centrale e piu IMPORTANTE!
    # distanze EMD dell ellissi
    dist_color_ell = numpy.asarray([emd_from_hist(hist_color[-1], h, matrix) for h in hists_colors[:, -1]])

    dist_sift = numpy.linalg.norm(hists_sift - hist_sift, axis=1)
    # portiamo tutto nel range si sift
    dist_color_ell[:, 0] = (dist_color_ell[:, 0] - numpy.min(dist_color_ell[:, 0])) / numpy.max(
        dist_color_ell[:, 0]) * (numpy.max(dist_sift) - numpy.min(dist_sift)) + numpy.min(dist_sift)
    dist_color_ell[:, 1] = (dist_color_ell[:, 1] - numpy.min(dist_color_ell[:, 1])) / numpy.max(
        dist_color_ell[:, 1]) * (numpy.max(dist_sift) - numpy.min(dist_sift)) + numpy.min(dist_sift)
    dist_color_ell[:, 2] = (dist_color_ell[:, 2] - numpy.min(dist_color_ell[:, 2])) / numpy.max(
        dist_color_ell[:, 2]) * (numpy.max(dist_sift) - numpy.min(dist_sift)) + numpy.min(dist_sift)

    dist = dist_sift + dist_color_ell[:, 0] / 3 + dist_color_ell[:, 1] / 3 + dist_color_ell[:, 2] / 3
    #prendo indice 1
    #0 sara lui stesso
    #prendiamo il piu simile e mostriamole
    index = numpy.argsort(dist)[1]
    fig,ax = pyplot.subplots(ncols=2)
    ax[0].imshow(cv2.imread(dictionary["img"])[...,::-1])
    ax[1].imshow(cv2.imread(data["img_data"][index]["img"])[...,::-1])
    print("{} {}".format(dictionary["img"],data["img_data"][index]["img"]))
    pyplot.show()
