"""
This script runs the prediction over the test set
Predictions are random. This file is useful only to show
how predictions should be formatted in a file.
"""
from os.path import join, basename, exists
import os
import cv2
from glob import glob
import numpy
from random import shuffle
from utils import *

models = ['hist_sift','hist_color','resnet50','vgg16','vgg19','resnet50_cl','vgg16_cl','vgg19_cl','inception_resnet_v2','inception_resnet_v2_cl']
# folder of the train corpus
corpus_folder = '../cv-unsupervised-similarity/data/'
corpus_list = [basename(f) for f in glob(join(corpus_folder, '**', '*.jpg'))]

# folder of the test set
test_folder = 'test'
test_list = [basename(f) for f in glob(join(test_folder, '*.jpg'))]

data = numpy.load("../bow_50_1500.npy").item()
data_test = numpy.load("../test_data.npy").item()

def predict(image, prediction_folder,model):
    """
    Function that performs image retrieval on an image.
    :param image: the test image filename.
    :param prediction_folder: the folder where to store predictions.
    :return: None
    """
    # perform whatever prediction
    shuffle(corpus_list)
    dictionary = None
    #print image
    for dictionary_t in data_test["img_data"]:
        #print dictionary_t["img"]
        if image in dictionary_t["img"]:
            dictionary = dictionary_t
    if dictionary is None:
        print "Error", "file {} not in the dict".format(image)
        return

    if model == 'hist_sift':
        dists = HistSift(dictionary)

    if model == 'hist_color':
        dists = HistColor(dictionary)

    if model == 'resnet50':
        dists = Resnet50(dictionary)

    if model == 'vgg16':
        dists = Vgg16(dictionary)

    if model == 'vgg19':
        dists = Vgg19(dictionary)

    if model == 'inception_resnet_v2':
        dists = InceptionResnetV2(dictionary)

    if model == 'resnet50_cl':
        dists = ResNet50Class(dictionary)

    if model == 'vgg16_cl':
        dists = Vgg16Class(dictionary)

    if model == 'vgg19_cl':
        dists = Vgg19Class(dictionary)

    if model == 'inception_resnet_v2_cl':
        dists = InceptionResnetV2Class(dictionary)


    indexes = numpy.argsort(dists)
    imgs = [data["img_data"][j]["img"].split('/')[-1] for j in indexes]

    # write to file
    with open(join(prediction_folder, image + '.txt'), mode='w') as f:
        f.write(('{}\n'*len(imgs[1:])).format(*imgs[1:]))


# entry point
if __name__ == '__main__':
    # run predictions
    for model in models:
        prediction_folder = 'predictions_'+model
        if not exists(prediction_folder):
            os.makedirs(prediction_folder)
        print 'working with {}'.format(model)
        for image in test_list:
            predict(image, prediction_folder,model)
