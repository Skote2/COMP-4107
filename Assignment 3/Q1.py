#!/usr/bin/env python
 # Carleton University
 # Neural Networks
 # COMP 4107A
 # Fall 2018
 # 
 # Assignment 3
 # 
 # David N. Zilio
 # Aidan Crowther
 #
 # Question 1
 # Using the scikit-learn util to load MNIST written numbereric characters.
 # Implementing a Hopfield network to classify the image data for handwritten digits. specifically 1 and 5
 # 
 # Document the classification accuracy as a function of the number of images used to train the network
 #
 # BONUS: Improvements to basic hopfield traning from Storykey 97 article and contrast to original net
 #
 # NOTE some of the code used was taken from the recommended source by J.Loeber as it provides a funcional place to start. Same with the srowhani repository in git for some components of solutions and scikit learn base code


## Imports ##

from __future__ import division
import numpy as np
import random
import tensorflow as tf
import time
from math import floor

def flatten(grid):
    out = np.matrix(grid)
    return np.asarray(out.flatten(), dtype=np.int32)[0]

def toImage(inMatrix, labeled = True):
    result = ""
    if labeled:
        result = "Labled: " + str(inMatrix[1]) + '\n'
        inMatrix = inMatrix[0]
    dim = 28

    for i in range(0, dim):
        for j in range(0, dim):
            if(inMatrix[(i*dim) + j] == -1): result += ' .'
            else: result += '##'
        result += '\n'
    return result

def setSign (digit):
    for i in range(digit.size):
        digit[i] = 1 if digit[i] > 0 else -1
    return digit

class HopfieldNetwork(object):
    def hebbian(self):
        self.W = np.zeros([self.num_neurons, self.num_neurons])
        for image_vector, _ in self.train_dataset:
            self.W += np.outer(image_vector, image_vector) / self.num_neurons
        np.fill_diagonal(self.W, 0)

    def storkey(self):
        self.W = np.zeros([self.num_neurons, self.num_neurons])

        for image_vector, _ in self.train_dataset:
            self.W += np.outer(image_vector, image_vector) / self.num_neurons
            net = np.dot(self.W, image_vector)

            pre = np.outer(image_vector, net)
            post = np.outer(net, image_vector)

            self.W -= np.add(pre, post) / self.num_neurons
        np.fill_diagonal(self.W, 0)

    def __init__(self, train_dataset=[], mode='hebbian'):
        self.train_dataset = train_dataset
        self.num_training = len(self.train_dataset)
        self.num_neurons = len(self.train_dataset[0][0])

        self._modes = {
            "hebbian": self.hebbian,
            "storkey": self.storkey
        }

        self._modes[mode]()

    def activate(self, vector):
        changed = True
        while changed:
            changed = False
            indices = list(range(0, len(vector)))
            random.shuffle(indices)

            # Vector to contain updated neuron activations on next iteration
            new_vector = [0] * len(vector)

            for i in range(0, len(vector)):
                neuron_index = indices.pop()

                s = self.compute_sum(vector, neuron_index)
                new_vector[neuron_index] = 1 if s >= 0 else -1
                changed = not np.allclose(vector[neuron_index], new_vector[neuron_index], atol=1e-3)

            vector = new_vector

        return vector

    def compute_sum(self, vector, neuron_index):
        s = 0
        for pixel_index in range(len(vector)):
            pixel = vector[pixel_index]
            if pixel > 0:
                s += self.W[neuron_index][pixel_index]

        return s

################
# Doing things #
################
start = time.time()

mnist = tf.keras.datasets.mnist.load_data()
mnistTrain = mnist[0]
mnistTest = mnist[1]
train = []
test = []

trainingSize = 10
count = 0

for i in range(len(mnistTest[0])):
    label = mnistTest[1][i]
    if (label == 1 or label == 5):
        test.append((setSign(flatten(mnistTest[0][i])), label))
        count += 1
        if (count >= 20):
            break

count = 0

for s in range(1, 10):
    for i in range(len(mnistTrain[0])):
        label = mnistTrain[1][i]
        if (label == 1 or label == 5):
            train.append((setSign(flatten(mnistTrain[0][i])), label))
            count += 1
            if (count >= trainingSize*s):
                break

    hebbianNet = HopfieldNetwork(train_dataset=train, mode="hebbian")
    # storkeyNet = HopfieldNetwork(train_dataset=train, mode="storkey")

    print("Time to train: ", time.time() - start)
    start = time.time()

    minimum = []
    for t in test:
        out = hebbianNet.activate(t[0])
        norm = np.linalg.norm(t[0] - out)
        print (norm)
        print (toImage(t))
        print (toImage(out, labeled=False))
        if (norm==0):
            minimum.append(t)
    
    accuracy = 0
    # for t in test:
    #     out = hebbianNet.activate(t[0])
    #     for m in minimum:
    #         if (m[0] == out):
    #             if (t[1] == m[1]):
    #                 accuracy += 1
    #                 break
    accuracy = accuracy / len(test)

    print ("For hebbian network trained on: ", trainingSize*s, " images, an accuracy of: ", accuracy, " was obtained on a P/N of: ", (trainingSize*s/784))
