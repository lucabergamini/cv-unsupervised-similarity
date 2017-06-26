import numpy
from pyemd.emd import emd
import cv2
import scipy
from scipy import spatial

#data = numpy.load("../cv-unsupervised-similarity/bow_50_1500.npy").item()
data = numpy.load("../bow_50_1500.npy").item()

#prendo tutti istogrammi per ricerche
hists_sift = numpy.asarray([i["hist_sift"] for i in data["img_data"]])
hists_colors = numpy.asarray([i["hist_color"] for i in data["img_data"]])
resnet_feature = numpy.squeeze(numpy.asarray([i["resnet50"] for i in data["img_data"]]))
vgg16_feature = numpy.squeeze(numpy.asarray([i["vgg16"] for i in data["img_data"]]))
vgg19_feature = numpy.squeeze(numpy.asarray([i["vgg19"] for i in data["img_data"]]))
resnet_class_feature = numpy.squeeze(numpy.asarray([i["resnet50_cl"] for i in data["img_data"]]))
vgg16_class_feature = numpy.squeeze(numpy.asarray([i["vgg16_cl"] for i in data["img_data"]]))
vgg19_class_feature = numpy.squeeze(numpy.asarray([i["vgg19_cl"] for i in data["img_data"]]))

def get_emd_matrix(bins):
    """
    genera matrice dato il numero di bins
    :param bins: numero bin hist
    :param method: funzione numpy da applicare
    :return: matrice
    """
    matrix = numpy.zeros((bins,bins))
    for i in xrange(bins):
        matrix[i,i:] = numpy.arange(0,bins-i)
        matrix[i,0:i] = numpy.abs(numpy.arange(-i,0))

    return matrix

def emd_from_hist(hist_1, hist_2,matrix):
    """
    earth mover distances tra due istogrammi BGR
    se matrice a 1 viene simile a batacharia
    :param hist_1:
    :param hist_2:
    :param matrix: matrice simmetrica distanze
    :return: 3 distanze(una per ogni canale colore)
    """
    assert hist_1.shape == hist_2.shape
    # matrice simmetrica per le distanze
    # e il costo di spostare dal bin i al bin j
    matrix = matrix.astype("float64")
    emds = numpy.zeros(hist_1.shape[0])
    for i in numpy.arange(len(emds)):
        emds[i] = emd(hist_1[i].astype("float64"), hist_2[i].astype("float64"), matrix)
    return emds

matrix = get_emd_matrix(bins=8)
def HistSift(dictionary):
    feat = dictionary["hist_sift"]
    feats = hists_sift
    #dists = numpy.linalg.norm(feat - feats, axis=1)
    dists = []
    for f in feats:
        dists.append(numpy.abs(spatial.distance.cosine(feat, f)))
    dists = numpy.asarray(dists)
    return dists

def HistColor(dictionary):
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
    return dists

def Resnet50(dictionary):
    feat = dictionary["resnet50"]
    feats = resnet_feature
    dists = numpy.linalg.norm(feat - feats, axis=-1)
    return dists

def Vgg19(dictionary):
    feat = dictionary["vgg19"]
    feats = vgg19_feature
    dists = numpy.linalg.norm(feat - feats, axis=-1)
    return dists

def Vgg16(dictionary):
    feat = dictionary["vgg16"]
    feats = vgg16_feature
    dists = numpy.linalg.norm(feat - feats, axis=-1)
    return dists

def Vgg19(dictionary):
    feat = dictionary["vgg19"]
    feats = vgg19_feature
    dists = numpy.linalg.norm(feat - feats, axis=-1)
    return dists

def ResNet50Class(dictionary):
    feat = dictionary["resnet50_cl"]
    feats = resnet_class_feature

    # prendo 5 indice piu alto
    indexes_class = numpy.squeeze(numpy.argsort(feat, axis=-1))[-5:]
    # fill distanze
    # TODO qualcuno sistemi questo vi prego
    dists = 1000 + numpy.linalg.norm(feat - feats, axis=-1)

    # non ho trovato un modo di farlo senza for
    for j in xrange(len(dists)):
        if numpy.argmax(feats[j]) in indexes_class:
            dists[j] = numpy.linalg.norm(feat - feats[j:j + 1])
    return dists

def Vgg16Class(dictionary):
    feat = dictionary["vgg16_cl"]
    feats = vgg16_class_feature

    # prendo 5 indice piu alto
    indexes_class = numpy.squeeze(numpy.argsort(feat, axis=-1))[-5:]
    # fill distanze
    # TODO qualcuno sistemi questo vi prego
    dists = 1000 + numpy.linalg.norm(feat - feats, axis=-1)

    # non ho trovato un modo di farlo senza for
    for j in xrange(len(dists)):
        if numpy.argmax(feats[j]) in indexes_class:
            dists[j] = numpy.linalg.norm(feat - feats[j:j + 1])
    return dists

def Vgg19Class(dictionary):
    feat = dictionary["vgg19_cl"]
    feats = vgg19_class_feature

    # prendo 5 indice piu alto
    indexes_class = numpy.squeeze(numpy.argsort(feat, axis=-1))[-5:]
    # fill distanze
    # TODO qualcuno sistemi questo vi prego
    dists = 1000 + numpy.linalg.norm(feat - feats, axis=-1)

    # non ho trovato un modo di farlo senza for
    for j in xrange(len(dists)):
        if numpy.argmax(feats[j]) in indexes_class:
            dists[j] = numpy.linalg.norm(feat - feats[j:j + 1])
    return dists




def get_color_hist(img, bins, mode="bgr", normalize_channel=False, mask=None):
    """
    hist BGR o HSV concatenato(in questo ordine) con bin per ogni canale
    :param img: immagine BGR
    :param mode: se BGR o HSV
    :param bins: numero bin di ogni canale
    :param normalize_channel: se normalizzare il canale
    :return: hist, se bin= 40->40*3 = 120
    """

    h_area = numpy.zeros((3, bins), dtype="float32")
    assert mode in ["bgr", "hsv"]
    if mode == "bgr":
        for i in numpy.arange(3):
            h = cv2.calcHist([img], channels=[i], histSize=[bins], mask=mask, ranges=[0, 256])
            if normalize_channel:
                h /= numpy.sum(h)
            h_area[i] = numpy.squeeze(h)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for i in numpy.arange(3):
            if i == 0:
                h = cv2.calcHist([img], channels=[i], histSize=[bins], mask=mask, ranges=[0, 180])
            else:
                h = cv2.calcHist([img], channels=[i], histSize=[bins], mask=mask, ranges=[0, 256])

            if normalize_channel:
                h /= numpy.sum(h)
            h_area[i] = numpy.squeeze(h)
        return h_area

