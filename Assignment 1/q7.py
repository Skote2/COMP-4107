from numpy import *
from scipy.linalg import *

import numpy as np
import scipy.linalg as sp
import csv
import math

database = ''

with open("MovieLens/ratings.csv", mode="r") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    results = []

    userMax = 0
    movieMax = 0

    for elem in csv_reader:
        results.append(elem)
        if(int(elem['userId']) > userMax): userMax = int(elem['userId'])
        if(int(elem['movieId']) > movieMax): movieMax = int(elem['movieId'])

    database = np.tile(0, (userMax, movieMax))

    for result in results:
        database[int(result['userId'])-1, int(result['movieId'])-1] = float(result['rating'])

tempDB = []

for i in range(0, len(database[:, 0])):
    if((database[i] != 0).sum(0) > 20): tempDB.append(list(database[i]))

database = matrix(tempDB)
np.random.shuffle(database)

trainingData = database[0 : math.floor(len(database)*0.8)]
testingData = database[math.floor(len(database)*0.8) : len(database)]

def modifySVD(addMatrix, U, S, V):
    addPrime = np.dot(addMatrix, np.dot(V.T, inv(S)))
    U = vstack((U, addPrime))
    return U

def getColAvg(index):
    avg = 0
    count = 0
    for j in testingData:
        iteration = 0
        for i in np.nditer(testingData[j]):
            if(i != 0 and iteration == index): 
                avg += i
                count += 1
                break
            iteration += 1
    return avg/count

def getRowAvg(index):
    avg = 0
    count = 0
    for i in np.nditer(testingData[index]):
        if(i != 0): 
            avg += i
            count += 1
    return avg/count

def calcPred(row, col, U, S, V):
    rowAvg = getRowAvg(row)
    colAvg = getColAvg(col)

    return rowAvg + np.dot(np.dot(np.dot(U, sp.sqrtm(S).T), rowAvg), np.dot(np.dot(sp.sqrtm(S), V.T), colAvg))

inputMatrix = matrix(trainingData[0:50, :])

U, S, V = np.linalg.svd(inputMatrix, full_matrices=False)
S = np.diag(S)

for i in range(1, len(trainingData)//50+1):
    toAdd = ''
    if(i*50+50 < len(trainingData)): toAdd = matrix(trainingData[i*50:i*50+50, :])
    else: toAdd = matrix(trainingData[i*50:len(trainingData), :])

    U = modifySVD(toAdd, U, S, V)