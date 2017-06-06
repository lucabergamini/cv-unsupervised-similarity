from utils import resnet50_embedding, prepare_image

import cv2

import glob
import pickle
import gzip

if __name__ == "__main__":
    model = resnet50_embedding()

    result = []

    dataset = glob.glob('dataset/Fotografie/*/*')
    for filename in dataset:
        try:
            print 'Reading {}'.format(filename)
            img = prepare_image(cv2.imread(filename))
            feats = model.predict(img)
            result.append([filename, feats])
        except:
            print 'Error processing {}'.format(filename)

    pickle.dump(result, gzip.open('resnet50_feats.pklz', 'wb'))
