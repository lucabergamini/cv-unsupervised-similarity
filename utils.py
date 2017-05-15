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
    for i in xrange(3):
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
    for i in xrange(len(emds)):
        emds[i] = emd(hist_1[i].astype("float64"), hist_2[i].astype("float64"), matrix)
    return emds

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