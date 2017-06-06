from utils import resnet50_embedding, prepare_image, show_results

import numpy as np
import cv2

import pickle
import gzip

if __name__ == "__main__":
    model = resnet50_embedding()

    img = cv2.imread('test/goat.jpg')
    feats = model.predict(prepare_image(img)).reshape((-1,))

    dataset = pickle.load(gzip.open('resnet50_feats.pklz', 'rb'))
    dataset_names = np.array([sample[0] for sample in dataset])
    dataset_feats = np.array([sample[1].reshape((-1,)) for sample in dataset])

    dists = np.linalg.norm(dataset_feats - feats, axis=1)

    top8 = np.argsort(dists)[0:8]
    top8_names = dataset_names[top8]
    top8_imgs = [cv2.imread(f) for f in top8_names]

    show_results(img, top8_imgs)
