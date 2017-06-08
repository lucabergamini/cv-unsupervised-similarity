import pickle
import gzip
import numpy as np

# Use this if you have your dataset in a different folder and
# want to fix the filenames in the features dicts.

if __name__ == "__main__":
    dict = np.load('bow_50_1500.npy').item()

    vgg16_feats = pickle.load(gzip.open('convolutional/vgg16_feats.pklz', 'rb'))
    vgg19_feats = pickle.load(gzip.open('convolutional/vgg19_feats.pklz', 'rb'))
    resnet50_feats = pickle.load(gzip.open('convolutional/resnet50_feats.pklz', 'rb'))

    img_data = dict['img_data']
    for i, sample in enumerate(img_data):

        vgg16 = [s for s in vgg16_feats if s[0].replace('dataset/Fotografie', 'data', 1) == sample['img']][0]
        vgg19 = [s for s in vgg19_feats if s[0].replace('dataset/Fotografie', 'data', 1) == sample['img']][0]
        resnet50 = [s for s in resnet50_feats if s[0].replace('dataset/Fotografie', 'data', 1) == sample['img']][0]

        img_data[i]['vgg16'] = vgg16[1]
        img_data[i]['vgg19'] = vgg19[1]
        img_data[i]['resnet50'] = resnet50[1]

    np.save('bow_50_1500.npy', dict)
