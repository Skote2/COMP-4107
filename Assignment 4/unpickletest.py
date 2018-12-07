import numpy as np
import pickle
import matplotlib.pyplot as plt
import random

def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def loadBatch(folder, id):
    with open(folder + '/data_batch_' + str(id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return normalize(features), labels

def loadLabels():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def display(path, batch, id):
    features, labels = loadBatch(path, batch)

    if not (0 <= id < len(features)):
        print('{} samples in batch {}.  {} is out of range.'.format(len(features), batch, id))
        return None

    print('\nStats of batch #{}:'.format(batch))
    print('# of Samples: {}\n'.format(len(features)))

    labelNames = loadLabels()
    sampleImage = features[id]
    sampleLabel = labels[id]

    print('\nExample of Image {}:'.format(id))
    print('Image - Min Value: {} Max Value: {}'.format(sampleImage.min(), sampleImage.max()))
    print('Image - Shape: {}'.format(sampleImage.shape))
    print('Label - Label Id: {} Name: {}'.format(sampleLabel, labelNames[sampleLabel]))
    plt.imshow(sampleImage, interpolation='nearest')
    plt.show()

display("CIFARdata", 1, 1)