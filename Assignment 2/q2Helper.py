import matplotlib.pyplot as plt
from math import floor
from numpy import mat
from numpy import asarray
from random import random
from random import randint
from copy import deepcopy
import time

class timer:
    startTime = 0.0
    stopTime = 0.0

    def start(self):
        self.startTime = time.time()

    def stop(self):
        self.stopTime = time.time()

    def __str__(self):
        if (self.startTime >= self.stopTime):
            self.stopTime = time.time()
        return "{:.3f}".format(self.stopTime - self.startTime)

def load(fileName):
    data = []

    fileReader = open(fileName, "r")

    grid = []
    for line in fileReader:
        if (line == "\n"):
            data.append(grid)
            grid = []
        else:
            for char in line:
                if (char != "\n"):
                    grid.append(char)
    if (grid != []):
        data.append(grid)
    
    return mat(data)

#
def chart(xSet, ySet, labels):
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')
    plt.xlabel("Noise Level")
    plt.ylabel("Percentage of recognition errors")

    for i in range(0, len(xSet)):
        plt.plot(xSet[i], ySet[i], marker='', color=palette(i), linewidth=1, alpha=0.9, label=labels[i])
    plt.legend()
    plt.show()

def chart2(xSet, ySet):
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')
    plt.xlabel("Epochs")
    plt.ylabel("Neural net training accuracy")

    plt.plot(xSet, ySet, marker='', color=palette(0), linewidth=1, alpha=0.9)
    plt.show()

# takes in a matrix of data to fuzz up by the specified noise ammount
# returns that data set with some noise added
def makeNoisy(data, noiseFrequency):
    data = deepcopy(data)
    noisyData = []
    for i in range(0, len(data)):               #for each image
        noisyData.append(asarray(data[i:i+1, :])[0:1][0])   # gets array datatype from matrix
        noisyPixels = floor(noiseFrequency)
        r = random()
        if (r < (noiseFrequency - floor(noiseFrequency))): # this is handling the decimal
            noisyPixels += 1
        fliped = [0]*noisyData[i].size          # keep track so synthetic noise isn't nulled by itself
        for j in range (0, noisyPixels):        # for each blured pixel
            noise = randint(0, len(noisyData[i])-1) # TODO fix python because this fucking function's broken and doesn't work
            while(fliped[noise] == 1):          # loop until it isn't taken
                noise = randint(0, len(noisyData[i])-1)
            fliped[noise] = 1
            if (noisyData[i][noise] == '1'):          # create noise
                noisyData[i][noise] = '0'
            else:
                noisyData[i][noise] = '1'

    return mat(noisyData)

def printImg(data):
    for i in range(0, 7):
        for j in range(0, 5):
            print(data[(i*5)+j], end="")
        print("", end="\n")