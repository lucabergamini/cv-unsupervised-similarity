from __future__ import print_function, division
import numpy
import cv2
from utils import *
from matplotlib import pyplot


#carico data
data = numpy.load("bow_50_1500.npy").item()
#prendo tutti istogrammi per ricerche
hists_sift = numpy.asarray([i["hist_sift"] for i in data["img_data"]])
hists_colors = numpy.asarray([i["hist_color"] for i in data["img_data"]])
resnet_feature = numpy.squeeze(numpy.asarray([i["resnet50"] for i in data["img_data"]]))
vgg16_feature = numpy.squeeze(numpy.asarray([i["vgg16"] for i in data["img_data"]]))
vgg19_feature = numpy.squeeze(numpy.asarray([i["vgg19"] for i in data["img_data"]]))
# matrice per EMD
matrix = get_emd_matrix(8)
choices = ["BOW", "COLOR_HIST", "RESNET", "VGG16", "VGG19"]
for i, c in enumerate(choices):
    print("{} per {}".format(i, c))
choice = int(raw_input(":"))
assert choice in xrange(0, len(choices))
# per ogni immagine
for dictionary in data["img_data"]:
    # estraggo feature giusta
    if choice == 0:
        feat = dictionary["hist_sift"]
        feats = hists_sift
        dists = numpy.linalg.norm(feat - feats, axis=1)

    elif choice == 1:
        feat = dictionary["hist_color"]
        feats = hists_colors
        # distanze dei vari corner pesata 0.25
        dists = None
        for j in xrange(4):
            if dists is None:
                dists = numpy.asarray([emd_from_hist(feat[j], h, matrix) for h in hists_colors[:, j]])
                # questo lavora sui canali indipendentemente
                # quindi faccio la norma delle norme
                dists = 0.25 * numpy.linalg.norm(dists, axis=-1)
            else:
                dists_t = numpy.asarray([emd_from_hist(feat[j], h, matrix) for h in hists_colors[:, j]])
                # questo lavora sui canali indipendentemente
                # quindi faccio la norma delle norme
                dists_t = 0.25 * numpy.linalg.norm(dists_t, axis=-1)
                dists += dists_t
        # distanza centrale
        dists_t = numpy.asarray([emd_from_hist(feat[-1], h, matrix) for h in hists_colors[:, -1]])
        # questo lavora sui canali indipendentemente
        # quindi faccio la norma delle norme
        dists_t = numpy.linalg.norm(dists_t, axis=-1)
        dists += dists_t


    elif choice == 2:
        feat = dictionary["resnet50"]
        feats = resnet_feature
        dists = numpy.linalg.norm(feat - feats, axis=1)

    elif choice == 3:
        feat = dictionary["vgg16"]
        feats = vgg16_feature
        dists = numpy.linalg.norm(feat - feats, axis=1)

    elif choice == 4:
        feat = dictionary["vgg19"]
        feats = vgg19_feature
        dists = numpy.linalg.norm(feat - feats, axis=1)

    indexes = numpy.argsort(dists)[0:9]
    imgs = [cv2.imread(data["img_data"][j]["img"]) for j in indexes]
    show_results(imgs[0], imgs[1:])
    pyplot.show()
