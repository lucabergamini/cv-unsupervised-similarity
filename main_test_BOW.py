from __future__ import print_function, division
import numpy
import cv2
from utils import *
from matplotlib import pyplot

#carico data
data = numpy.load("data.npy").item()

#prendo tutti istogrammi per ricerche
hists = numpy.asarray([i["hist_sift"] for i in data["img_data"]])
for dictionary in data["img_data"]:
    #prendo hist
    hist_1 = dictionary["hist_sift"]
    #adesso devo cercare i piu simili
    #0 sara lui stesso
    #prendiamo il piu simile e mostriamole
    index = numpy.argsort(numpy.linalg.norm(hists-hist_1,axis=1))[1]
    fig,ax = pyplot.subplots(ncols=2)
    ax[0].imshow(cv2.imread(dictionary["img"])[...,::-1])
    ax[1].imshow(cv2.imread(data["img_data"][index]["img"])[...,::-1])
    print("{} {}".format(dictionary["img"],data["img_data"][index]["img"]))
    pyplot.show()
