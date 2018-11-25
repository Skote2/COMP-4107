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
for netNum in range(0, 2):#for each net with hidden neuron numbers 5, 15, ... 25
    model.append(keras.Sequential())
    for i in range(0, len(options.layerSizes)): #add layers to the net
        model[netNum].add(keras.layers.Dense(options.layerSizes[i], 
                                             activation="sigmoid", 
                                             kernel_initializer=options.initializer))

    model[netNum].compile(optimizer=AdamOptimizer(0.001), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # ### TRAIN ###
#net trained on only clean
model[0].fit(cleanData, labels, epochs=500, batch_size=5, verbose=1)

# net trained on both types
for trainingCycles in range(0, 500):
    if (trainingCycles % 10 == 0):
        print("Training noise set, cycle: ", trainingCycles, "/500")
    noisyData = []
    for i in range (0, 7): # noise levels {0.0, 0.5, 1.0 ... 3.0}
        noisyData.append(helper.makeNoisy(cleanData, i/2))
    for i in range (0, 7): ## for all noisy and clean data
        model[1].fit(noisyData[i], noisyLabels[i], epochs=1, batch_size=5, verbose=0)

##########
## TEST ##
##########

# old labels are still going to co-respond but the noise should be new for testing data
noisyData = []
for i in range (0, 7): # noise levels {0.0, 0.5, 1.0 ... 3.0}
    noisyData.append(helper.makeNoisy(cleanData, i/2))

results = []
for netNum in range(0, 2):
    results.append([])
    for i in range (0, 7): ## for all levels of noise
        results[netNum].append(
            model[netNum].evaluate(noisyData[i], noisyLabels[i], batch_size=5)
        )

for netNum in range(0, 2):
    for noiseLvl in range (0, 7):
        print(model[netNum].metrics_names[0], ": ", "{:.3f}".format(results[netNum][noiseLvl][0]))
        print(model[netNum].metrics_names[1], ": ", "{:.3f}".format(results[netNum][noiseLvl][1]*100), '%')


print("\nThe whole thing took: ", timer, "s")

#########################################
# This is the start of graphing outputs #
#########################################
x = []
y = []
l = ["trained without noise", "trained with noise"]
for netNum in range(0, 2):
    x.append([])
    y.append([])
    for noiseLvl in range(0, 7):
        x[netNum].append(noiseLvl/2)
        y[netNum].append((1-results[netNum][noiseLvl][1])*100)

helper.chart(x, y, l)

print("")