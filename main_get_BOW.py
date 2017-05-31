from __future__ import print_function, division
from utils import *
import os

SIFT_NUMBER = 50
CLUSTER_NUMBER = 1500
COLOR_BINS = 8
COLOR_SPACE = "bgr"
#carico lista di immagini
BASE_FOLDER = "data/"
SUB_FOLDERS = [BASE_FOLDER+i+"/" for i in os.listdir(BASE_FOLDER)]
imgs_names = []
for i in SUB_FOLDERS:
    imgs_names.extend([i+j for j in os.listdir(i)])

#estraggo sift
data = {"vocabulary": None, "img_data": []}
for i,img_name in enumerate(imgs_names):
    img = cv2.imread(img_name)
    sift_t = get_sift(img, n_sift=SIFT_NUMBER, edge_t=10)
    if sift_t is None:
        print("no sift in {}".format(img_name))
    else:
        #posso aggiungerla
        dictionary = {"img": img_name, "sift": sift_t, "hist_sift": None}
        # devo aggiungere anche hist colore delle parti
        maskes = get_mask(img)
        hist_color = numpy.zeros((len(maskes), 3, COLOR_BINS))
        for j, mask in enumerate(maskes):
            h = get_color_hist(img, bins=COLOR_BINS, mode=COLOR_SPACE, normalize_channel=True, mask=mask)
            hist_color[j] = h
            dictionary["hist_color"] = hist_color

        data["img_data"].append(dictionary)
    if i % 100 == 0:
        print(i)
print("sift done")
sift_list = numpy.asarray([i["sift"] for i in data["img_data"]])

data["vocabulary"] = get_BOW_vocabulary(sift_list,cluster_number=CLUSTER_NUMBER)
for i in data["img_data"]:
    #ottengo hist
    i["hist_sift"] = get_BOW_hist(i["sift"],data["vocabulary"])

numpy.save("bow_{}_{}".format(SIFT_NUMBER,CLUSTER_NUMBER),data)

