 # Carleton University
 # Neural Networks
 # COMP 4107A
 # Fall 2018
 # 
 # Assignment 2
 # 
 # David N. Zilio
 # Aidan Crowther
 #
 # Question 2
 # Building a neural net to interpret 7x5px B&W "images" with noise

import q2Helper as helper
import numpy    as np
from tensorflow import keras
from tensorflow.train import AdamOptimizer
from math import floor

class options:
    inputFileName = "q2-patterns.txt"
    layerSizes = [35, 15, 31]
    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.15)
op = options()


timer = helper.timer()
timer.start()

# Prepping clean data & labels
cleanData = helper.load(options.inputFileName)
labels = []
for i in range (0, len(cleanData)):
    a = [0]*len(cleanData)
    a[i] = 1
    labels.append(a)
labels = np.mat(labels)

#prep the fuzzy data and co-respoding labels
noisyData = []
noisyLabels = []
for i in range (0, 7): # noise levels {0.0, 0.5, 1.0 ... 3.0}
    noisyData.append(helper.makeNoisy(cleanData, i/2))
    noisyLabels.append(labels)

##########################
## BUILD the Neural net ##
##########################
model = []
for netNum in range(0, 1):#for each net with hidden neuron numbers 5, 15, ... 25
    options.layerSizes[1] = 15
    model.append(keras.Sequential())
    for i in range(0, len(options.layerSizes)): #add layers to the net
        model[netNum].add(keras.layers.Dense(options.layerSizes[i], 
                                             activation="sigmoid", 
                                             kernel_initializer=options.initializer))

    model[netNum].compile(optimizer=AdamOptimizer(0.001), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # ### TRAIN ###

# Three step training
# 1. train on ideal data 
firstResults = model[0].fit(cleanData, labels, epochs=339, batch_size=1, verbose=1)
# 2. train on noisy for 10 epochs
for trainingCycles in range(0, 100):
    for netNum in range(0, 1):
        noisyData = []
        for i in range (0, 7): # noise levels {0.0, 0.5, 1.0 ... 3.0}
            noisyData.append(helper.makeNoisy(cleanData, i/2))
        for i in range (2, 7): ## for only noisy data
                model[netNum].fit(noisyData[i], noisyLabels[i], epochs=1, batch_size=5, verbose=0)
# 3. train on ideal data
lastResults = model[0].fit(cleanData, labels, epochs=299, batch_size=10, verbose=1)

##########
## TEST ##
##########

#no testing for this part

print("\nThe whole thing took: ", timer, "s")

#########################################
# This is the start of graphing outputs #
#########################################

x = firstResults.epoch
y = firstResults.history['acc']
helper.chart2(x, y)

x = lastResults.epoch
y = lastResults.history['acc']
helper.chart2(x, y)

print("")