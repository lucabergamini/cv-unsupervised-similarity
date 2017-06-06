import cv2
import numpy as np

import matplotlib.pyplot as plt

from vgg16 import VGG16
from vgg19 import VGG19

# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50

from keras.optimizers import SGD


def vgg16_embedding():
    model = VGG16('vgg16_weights.h5')
    # model = VGG16()

    # Get rid of the last classification and dropout layers

    # This works for Sequential model only
    # model.pop()
    # model.pop()

    model.layers.pop()
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model


def vgg19_embedding():
    model = VGG19('vgg19_weights.h5')
    # model = VGG19()

    # Get rid of the last classification and dropout layers

    # This works for Sequential model only
    # model.pop()
    # model.pop()

    model.layers.pop()
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model


def resnet50_embedding():
    model = ResNet50()

    # Get rid of the last fully connected layer

    # This works for Sequential model only
    # model.pop()
    # model.pop()

    model.layers.pop()
    # model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model


def prepare_image(img):
    """
    Prepares image to be used as input of VGG16/VGG19
    :param img: input image to be prepared
    :return: prepared image
    """
    img = cv2.resize(img, (224, 224)).astype(np.float32)
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    img = img.transpose((1, 0, 2))
    img = np.expand_dims(img, axis=0)
    return img


def show_results(img, results):
    """
    Plots the comparing image together with the results, 4 per row.
    :param img: the image to be compared with results
    :param results: images resulting from retrieval
    :return:
    """
    rows = int(np.ceil(len(results) / 4))

    plt.subplot2grid((rows, 6), (0, 0), rowspan=2, colspan=2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap='gray')

    for i in range(len(results)):
        plt.subplot2grid((rows, 6), (i / 4, 2 + (i % 4)))
        plt.imshow(cv2.cvtColor(results[i], cv2.COLOR_BGR2RGB), cmap='gray')

    plt.show()
