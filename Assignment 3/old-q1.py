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
 # NOTE some of the code used was taken from the recommended source by J.Loeber as it provides a funcional place to start.


## Imports ##

import matplotlib.pyplot as plt
import numpy as np
import time
import threading
from math import ceil
from math import floor
from sklearn.datasets import fetch_mldata
from random import shuffle


## Function Definitions ##

# graphically displays a 768x1 vector, representing a digit
def display_digit(digit, labeled = True, title = ""):
    if labeled:
        digit = digit[1]
    image = digit
    plt.figure()
    fig = plt.imshow(image.reshape(28,28))
    fig.set_cmap('gray_r')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    if title != "":
        plt.title("Inferred label: " + str(title))
    plt.draw()


#Print an input array into a human readable form
def toImage(inMatrix, labeled = True):
    result = ""
    if labeled:
        result = "Labled: " + str(inMatrix[0]) + '\n'
        inMatrix = inMatrix[1]
    dim = 28

    for i in range(0, dim):
        for j in range(0, dim):
            if(inMatrix[(i*dim) + j] == -1): result += ' .'
            else: result += '##'
        result += '\n'
    return result

def sgn (num):
    if (num >= 0):
            return 1
    return -1

def setSign (digit):
    for i in range(digit[1].size):
        if (digit[1][i]-128 >= 0):
            digit[1][i] = 1
        else:
            digit[1][i] = -1
    return digit

def train(training_data):
    dim = training_data[0][1].size
    weights = np.zeros([dim, dim])
    for data in training_data:
        weights += np.outer(data[1], data[1])
    weights -= len(training_data) * np.identity(dim)
    # for diag in range(dim):
    #     weights[diag][diag] = 0
    return weights

def correct(testData, weights):
    corrected = np.empty(testData[1].size, testData[1].dtype)
    for i in range(testData[1].size):
        sum = 0
        for j in range(testData[1].size):
            sum += weights[i][j]*testData[1][j]
        corrected[i] = sgn(sum)
    return (testData[0], corrected)


#https://github.com/ccd97/hello_nn/blob/master/Hopfield-Network/code/np_hnn_reconstruction.py
# Function to test the network -- this only works for noisy data, I would have to alter it to local minima
def test(weights, testing_data):
    success = 0.0

    output_data = []

    for data in testing_data:
        true_data = data[0]
        noisy_data = data[1]
        predicted_data = retrieve_pattern(weights, noisy_data)
        if np.array_equal(true_data, predicted_data):
            success += 1.0
        output_data.append([true_data, noisy_data, predicted_data])

    return (success / len(testing_data)), output_data

# Function to retrieve individual noisy patterns
def retrieve_pattern(weights, data, steps=10):
    res = np.array(data)

    for _ in range(steps):
        for i in range(len(res)):
            raw_v = np.dot(weights[i], res)
            if raw_v > 0:
                res[i] = 1
            else:
                res[i] = -1
    return res

#https://www.tutorialspoint.com/python/python_multithreading.htm
class myThread (threading.Thread):
    def __init__(self, threadID, name, item, itemNum):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.item = item
        self.itemNum = itemNum
    def run(self):
        # print ("Starting " + self.name)
        self.item = setSign(self.item)
        # print ("Exiting " + self.name)

# MULTITHREADING WAS SLOWER :'(
# live = True
# threadCount = 7
# threads = []
# itemNum = 0
# for i in range(threadCount):
#     threads.append(myThread(i, "Thread-%d" % i, digits[itemNum], itemNum))
#     itemNum += 1

# itemNum = 0
# while live:
#     live = False
#     for i in range(threadCount):
#         if (not threads[i].isAlive()):
#             if (itemNum < len(digits)):
#                 digits[threads[i].itemNum] = threads[i].item
#                 threads[i] = myThread(i, "Thread-%d" % i, digits[itemNum], itemNum)
#                 threads[i].start()
#                 itemNum += 1
#                 live = True
#         else:
#             live = True


###############
# Doing Stuff #
###############
start = time.time()

mnist = fetch_mldata('MNIST original')
digits = []

#set up tuples
for i in range(0, len(mnist.data)):
    label = mnist.target[i]
    if (label == 5):# or label == 5): # exclusively 1 and 5
        digits.append((label, mnist.data[i].astype(np.int32)))

print("Loading up the data took: %.2fs" % (time.time() - start))

# shuffle(digits)

for i in range(len(digits)):
    digits[i] = setSign(digits[i])

trainData = []
testingData = []
trainingPortion = 0.01
trainingPortion = floor(len(digits)*trainingPortion)
testingPortion = len(digits)-trainingPortion
for i in range(trainingPortion):
    trainData.append(digits[i])
for i in range(testingPortion):
    testingData.append(digits[i + trainingPortion])


print("# Train: ", len(trainData), "# Test:  ", len(testingData))
print("Handling the data took: %.2fs" % (time.time() - start))
start = time.time()

#number, in hundreds of images to train off of
trainingSize = 141
weights = train(trainData)
print("Training took: %.2fs" % (time.time() - start))
start = time.time()

#recursive testing funciton
def runTest (testItem, weight, limit=50, depth=0):
    corrected = correct(testItem, weight)
    #Exit recursion if:
    if (depth >= limit):# reached depth limit
        print("depth limit reached")
        return corrected
    norm = np.linalg.norm(testItem[1]-corrected[1])
    if (norm == 0):# or no changes were observed
        print("norm was equal to zero at depth: " + str(depth))
        return corrected
    if (depth % 10 == 0):
        toImage(corrected)

    return runTest(corrected, weight, limit, depth+1)

for t in testingData:
    print(toImage(runTest(test, weights, 20)))
    print(toImage(test))
    time.sleep(0.5)
print("Network feedback took: %.2fs" % (time.time() - start))
start = time.time()
print()