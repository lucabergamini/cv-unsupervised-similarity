from __future__ import print_function, division
import cv2
#from skimage.feature import hog
import numpy
from pyemd.emd import emd


def get_mask(img):
    """
    get mask for the image
    :param img: 
    :return: list of maskes
    """
    maskes = []
    # convert the image to the HSV color space and initialize
    # the features used to quantify the image

    # grab the dimensions and compute the center of the image
    (h, w) = img.shape[:2]
    (cX, cY) = (int(w * 0.5), int(h * 0.5))

    # divide the image into four rectangles/segments (top-left,
    # top-right, bottom-right, bottom-left)
    segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
                (0, cX, cY, h)]

    # construct an elliptical mask representing the center of the
    # image
    (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
    ellipMask = numpy.zeros(img.shape[:2], dtype="uint8")
    cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

    # loop over the segments
    for (startX, endX, startY, endY) in segments:
        # construct a mask for each corner of the image, subtracting
        # the elliptical center from it
        cornerMask = numpy.zeros(img.shape[:2], dtype="uint8")
        cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
        cornerMask = cv2.subtract(cornerMask, ellipMask)
        maskes.append(cornerMask)
    maskes.append(ellipMask)
    return maskes


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


def get_sift(img, n_sift, edge_t=5):
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
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_sift, nOctaveLayers=5, contrastThreshold=0.08, edgeThreshold=edge_t,
                                       sigma=1.6)

    keypoints, descriptor = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask=None)
    return descriptor


def sift_match(features_1, features_2):
    """
    2NN-sift-match tra due immagini usando criterio del paper
    :param features_1: features immagine 1 segnata come query
    :param features_2: features immagine 2 segnata come train(non si sa perche)
    :return: numpy array con indici e distanze di quei match che superano la soglia
    """
    # distanza L2
    bf = cv2.BFMatcher(crossCheck=False)
    # due match per ognunno
    matches = bf.knnMatch(features_1, features_2, k=2)
    matches_array = []
    for match_1, match_2 in matches:
        if match_1.distance >= 0.75 * match_2.distance:
            continue
        # trovato match
        matches_array.append((match_1.queryIdx, match_1.trainIdx, match_1.distance))
    # torno come float per distanza,ma gli indici vogliono int
    return numpy.asarray(matches_array, dtype="float32")


def get_BOW_vocabulary(sift_features, cluster_number):
    """
    date le features di un training set le aggrega in cluster_number cluster(k-means) e resituisce i centroidi
    perche ho fatto questa funzione? bella domanda mio giovane padawan
    il motivo risiede nella lentezza del calcolo dei centri cluster
    il vocabolario puo essere salvato cosi da chiamare questa funzione solo una volta
    :param sift_features: feature sift come numpy array(n_feature,128)
    :param cluster_number: numero cluster
    :return: vocabolario, cioe array(cluster_number,128)
    """

    bow = cv2.BOWKMeansTrainer(clusterCount=cluster_number)
    for sift in sift_features:
        bow.add(sift)
    vocabulary = bow.cluster()
    return vocabulary


def get_BOW_hist(sift_features, vocabulary):
    """
    histogramma delle feature sift di una immagine usando la distanza con il vocabulary
    :param sift_features: vettore features di una immagine
    :param vocabulary: vocabolario features
    :return: histogramma array(cluster_number)
    """
    hist = numpy.zeros(len(vocabulary), dtype="float32")
    for sift in sift_features:
        # calcolo distanze euclide con vocabolario
        dist = numpy.linalg.norm(vocabulary - sift, axis=1)
        # assegno alla minore la sift incrementando hist
        hist[numpy.argmin(dist)] += 1
    # normalizzo dividendo per numero di sift
    hist /= len(sift_features)
    return hist

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
