import pickle
import gzip
import numpy as np

# Use this if you have your dataset in a different folder and
# want to fix the filenames in the features dicts.

if __name__ == "__main__":
    dict = np.load('bow_50_1500.npy').item()

    vgg16_classes = pickle.load(gzip.open('convolutional/vgg16_classes.pklz', 'rb'))

    img_data = dict['img_data']
    for i, sample in enumerate(img_data):

        vgg16 = [s for s in vgg16_classes if s[0].replace('dataset/Fotografie', 'data', 1) == sample['img']][0]

        img_data[i]['vgg16_cl'] = vgg16[1]

    np.save('bow_50_1500.npy', dict)
