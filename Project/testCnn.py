import os
import tensorflow as tf
import numpy as np
import scipy as sp
import pickle
import matplotlib.pyplot as plt
import random

import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import ConvLSTM2D, LSTM, Reshape
from keras.models import Sequential
from keras.callbacks import EarlyStopping

earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=6, verbose=1, mode='auto')
callbacks_list = [earlystop]

dim1 = 0
dim2 = 0
trainingData, trainingLabels, testingData, testingLabels = None, None, None, None

def load(batchId):
    
    if(batchId == 0):
        with open('Data/TestingData.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        with open('Data/TrainingData'+str(batchId)+'.pkl', 'rb') as f:
            return pickle.load(f)

def getBatch(batchId):

    batch = load(batchId)
    batchOut = []
    labelsOut = []

    keys =  list(batch.keys())
    random.shuffle(keys)

    for key in keys:
        #batchOut.append(batch[key]['data'])
        #labelsOut.append(batch[key]['genre'])
        #'''
        for i in range(7):
            batchOut.append(batch[key]['data'][:, (i*67) : (((i+1)*67)-1)])
            labelsOut.append(batch[key]['genre'])
        #'''

    batchOut, labelsOut = np.array(batchOut), np.array(labelsOut)

    dim1, dim2 = batchOut[0].shape
    batchOut = batchOut.reshape(batchOut.shape[0], dim1, dim2, 1)

    return batchOut, labelsOut, dim1, dim2

def prepData():

    global trainingData
    global trainingLabels
    global testingData
    global testingLabels
    global dim1
    global dim2

    trainingData, trainingLabels, dim1, dim2 = getBatch(1)

    for i in range(2, 10):
        tempTrain, tempLabels, _, _ = getBatch(i)
        trainingData = np.concatenate((trainingData, tempTrain))
        trainingLabels = np.concatenate((trainingLabels, tempLabels))

    testingData, testingLabels, _, _ = getBatch(0)

    return

prepData()

#Stolen booty laieth below

num_classes = 8

model = Sequential()
model.add(Conv2D(8, kernel_size=(4, 4), strides=(1, 1),
                 activation='relu',
                 input_shape=(dim1, dim2, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(Conv2D(64, (2, 2), activation='relu'))

model.add(Reshape(target_shape=(48, 64)))
ConvLSTM2D(filters=8, kernel_size=(3, 3), input_shape=(None, 110, 64), padding='same', return_sequences=True,  stateful = True)
model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(110, 64)))
model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(trainingData, trainingLabels,
          batch_size=128,
          epochs=20,
          verbose=1,
          validation_data=(testingData, testingLabels),
          shuffle = True,
          callbacks=callbacks_list)

probabilities = model.predict(trainingData)

def print_probs(ps):
    for p, i in sorted([(p, i) for i, p in enumerate(ps)], reverse=True):
        print('{}: {:.4f}'.format(i, p))

for i in range(10):
    print("Guess #",i,": ")
    selected = random.randint(0, 5550)
    print_probs(probabilities[selected])
    print("Actual: ", trainingLabels[selected])