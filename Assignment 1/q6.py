from numpy import *
from scipy.linalg import *

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import time

from mlxtend.data import loadlocal_mnist

#Import datasets
data, labels = loadlocal_mnist(images_path="train-images-idx3-ubyte", labels_path="train-labels-idx1-ubyte")
testData, testLabels = loadlocal_mnist(images_path="t10k-images-idx3-ubyte", labels_path="t10k-labels-idx1-ubyte")

#Get starting point to determine runtime
start = int(round(time.time() * 1000))

#Print database dimensions
print('Dimensions: %s x %s' % (data.shape[0], data.shape[1]))
print('Class distribution: %s' % np.bincount(labels))

#Convert the input array into a 28x28 matrix
def reform(inMatrix, dim):
    result = np.tile(0, (dim, dim))
    for i in range(0, dim):
        for j in range(0, dim):
            result[i, j] = inMatrix[(dim*i)+j]
    return result
            
#Xonvert an input matrix into it's column vector form
def columnize(inMatrix, dim):
    result = np.arange(dim**2)
    for i in range(0, dim):
        for j in range(0, dim):
            result[(dim*j)+i] = inMatrix[i, j]
    return result

#Undo the columnization process, returning the original input array
def deColumnize(inMatrix, dim):
    result = np.arange(dim**2)
    for i in range(0, dim):
        for j in range(0, dim):
            result[(dim*i)+j] = inMatrix[(dim*j)+i]
    return result

#Print an input array into a human readable form
def toImage(inMatrix, dim):
    inMatrix = reform(inMatrix, dim)
    result = ""
    for i in range(0, dim):
        for j in range(0, dim):
            if(inMatrix[i, j] == 0): result += '.'
            else: result += '#'
        result += '\n'
    return result

#Add a column to a table
def addColumn(inMatrix, inColumn, index):
    for i in range(0, inMatrix.shape[0]):
        inMatrix[i, index] = inColumn[i]
    return inMatrix

#Convert output to a range from 0-1
def translate(val, minimum, maximum):
    valRange = maximum-minimum
    scaled = float(val - minimum)/float(valRange)
    return scaled

#Setup plot
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')
# plt.plot([0, 1, 2, 3], [4, 5, 6, 7], marker='', color=palette(0), linewidth=1, alpha=0.9, label=0)
# plt.plot([0, 1, 2, 3], [3, 4, 5, 6], marker='', color=palette(1), linewidth=1, alpha=0.9, label=0)
plt.xlabel("# of Basis Images")
plt.ylabel("Classification Percentage")

#Prepare a dictionary for results
values = {}
for j in range (0, 10):
    values[j] = np.tile(0, (data.shape[1], np.bincount(labels)[j]))

#Collect all images by their label, reforming into a m^2xn matrix
for j in range(0, 10):
    column = 0
    for i in range(0, len(data[:, 0])):
        if(labels[i] == j):
            addColumn(values[j], columnize(reform(data[i], 28), 28), column)
            column += 1

print("Tables generated in: ", int(round(time.time() * 1000))-start)
start = int(round(time.time() * 1000))

#Setup S, V, D tables to reduce processing load
def evaluatePerc(basis):
    tables = []
    for i in range(0, 10):
        val = U, S, V = np.linalg.svd(values[i][:, 0:basis], False)
        tables.append(val)

    id768 = np.identity(784)
    rtnVal = []
    #Iterate over all test images
    for i in range(0, 500):#testData.shape[0]):
        toUse = columnize(reform(testData[i], 28), 28)
        results = []

        n = 0
        v = 10000
        #Get its results for the residual
        for j in range(0, 10):
            #U, S, V = np.linalg.svd(values[j][:, 0:25])
            val = np.linalg.norm(np.dot((id768 - np.dot(tables[j][0], tables[j][0].T)), toUse), 2)
            results.append(val)
            if (val < v):
                n = j
                v = val
        #Append residual data to the plot
        #plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], results, marker='', color=palette(i), linewidth=1, alpha=0.9, label=0)
        if (n == testLabels[i]):
            rtnVal.append(True)
        else:
            rtnVal.append(False)

    return rtnVal

bases = []
percentile = []
print("Testing on Basis:")
for i in range(1, 51):
    if (i%2 == 0):    
        l = evaluatePerc(i)
        numTests = 0
        correct = 0
        for j in range(0,len(l)):
            numTests += 1
            if (l[j]):
                correct += 1
        bases.append(i)
        percentile.append(correct/numTests)
        print(i, " time: ", int(round(time.time() * 1000))-start)
    
plt.plot(bases, percentile, marker='', color=palette(i), linewidth=1, alpha=0.9, label=0)

#Show runtime and display the plot
print("Ploted in: ", int(round(time.time() * 1000))-start)
plt.show()