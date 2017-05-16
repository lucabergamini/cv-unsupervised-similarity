from __future__ import print_function, division
import cv2
from skimage.feature import hog
import numpy
from pyemd.emd import emd


def get_bgr_hist(img, bins, normalize_channel=False):
    """
    hist BGR concatenato(in questo ordine) con bin per ogni canale
    :param img: immagine BGR
    :param bins: numero bin di ogni canale
    :param normalize_channel: se normalizzare il canale
    :return: hist, se bin= 40->40*3 = 120
    """
    h_final = numpy.zeros((3, bins), dtype="float32")
    for i in numpy.arange(3):
        h = cv2.calcHist([img], channels=[i], histSize=[bins], mask=None, ranges=[0, 256])
        if normalize_channel:
            h /= numpy.sum(h)
        h_final[i] = numpy.squeeze(h)
    return h_final


def emd_from_hist(hist_1, hist_2):
    """
    earth mover distances tra due istogrammi BGR
    se matrice a 1 viene simile a batacharia
    :param hist_1:
    :param hist_2:
    :return: 3 distanze(una per ogni canale colore)
    """
    assert hist_1.shape == hist_2.shape
    # matrice simmetrica per le distanze
    # e il costo di spostare dal bin i al bin j
    # TODO definiscila!
    matrix = numpy.ones((hist_1.shape[1], hist_1.shape[1]), dtype="float64")
    emds = numpy.zeros(hist_1.shape[0])
    for i in numpy.arange(len(emds)):
        emds[i] = emd(hist_1[i].astype("float64"), hist_2[i].astype("float64"), matrix)
    return emds


def get_sift(img,n_sift,edge_t=5):
    """
    funzione di comodo cosi non diventiamo scemi con i parametri e con le chiamate
    ritorna n_sift descriptors(se li trova) usando edge_t come soglia per corner harris
    :param img: immagine
    :param n_sift: numero feature(ognuna 128 float)
    :param edge_t: soglia inversa(piu e bassa meno roba trova)
    :return:
    """
    # nfeatures 0 per avere tutto
    # posso scegliere le viste per ottava ma non il numero di ottave
    # contrast per prendere solo i massimi con un certo valore(espansione taylor)
    # edge per valore rapporto in Harris, se e alto uno dei due derivate e piu alta e quindi edge e non corner
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_sift, nOctaveLayers=3, contrastThreshold=0.08, edgeThreshold=edge_t,
                                       sigma=1.6)

    keypoints,descriptor = sift.detectAndCompute(img,mask=None)
    return descriptor


def sift_match(features_1,features_2):
    """
    2NN-sift-match tra due immagini usando criterio del paper
    :param features_1: features immagine 1 segnata come query
    :param features_2: features immagine 2 segnata come train(non si sa perche)
    :return: numpy array con indici e distanze di quei match che superano la soglia
    """
    #distanza L2
    bf = cv2.BFMatcher(crossCheck=False)
    # due match per ognunno
    matches = bf.knnMatch(features_1, features_2, k=2)
    matches_array = []
    for match_1, match_2 in matches:
        if match_1.distance >= 0.75 * match_2.distance:
            continue
        #trovato match
        matches_array.append((match_1.queryIdx,match_1.trainIdx,match_1.distance))
    #torno come float per distanza,ma gli indici vogliono int
    return numpy.asarray(matches_array,dtype="float32")

    # def get_hog(img,cell_number):
#     """
#     hog di skimage
#     :param img: immagine BGR
#     :param cell_number: numero di celle sulle due dimensioni
#     :return:
#     """
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     pixel_per_cell = (img.shape[0]/cell_number[0],img.shape[1]/cell_number[1])
#     print img.shape
#     return hog(image=img,orientations=9,pixels_per_cell=pixel_per_cell,cells_per_block=(4,4),block_norm="L2",visualise=True)
