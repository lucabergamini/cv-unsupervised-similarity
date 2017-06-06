import pickle
import gzip

# Use this if you have your dataset in a different folder and
# want to fix the filenames in the features dicts.

if __name__ == "__main__":
    vgg16_feats = pickle.load(gzip.open('vgg16_feats.pklz', 'rb'))
    vgg19_feats = pickle.load(gzip.open('vgg19_feats.pklz', 'rb'))
    resnet50_feats = pickle.load(gzip.open('resnet50_feats.pklz', 'rb'))

    for sample in vgg16_feats:
        sample[0] = sample[0].replace('00_Dataset', 'dataset', 1)

    for sample in vgg19_feats:
        sample[0] = sample[0].replace('00_Dataset', 'dataset', 1)

    for sample in resnet50_feats:
        sample[0] = sample[0].replace('00_Dataset', 'dataset', 1)

    pickle.dump(vgg16_feats, gzip.open('vgg16_feats.pklz', 'wb'))
    pickle.dump(vgg19_feats, gzip.open('vgg19_feats.pklz', 'wb'))
    pickle.dump(resnet50_feats, gzip.open('resnet50_feats.pklz', 'wb'))
