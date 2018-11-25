from numpy import *

import numpy as np
import scipy.linalg

accuracy = 0.01
results = []

A = matrix([[1, 2, 3],
            [2, 3, 4],
            [4, 5, 6],
            [1, 1, 1]])

B = matrix([[1, 1, 1, 1]]).T

randGen = np.random.rand(3, 1)

def calcX(x): return np.dot(A.T, np.dot(A, x)) - np.dot(A.T, B)

count = 0

for epsilon in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5]:

    x = randGen
    count = 0

    while(np.linalg.norm(calcX(x), 2) > accuracy):
        x = x - np.dot(epsilon, calcX(x))
        count += 1
    
    results.append([epsilon, count, x])

for result in results:
    print("Result using ",result[0],": ")
    print("X = [",result[2][0, 0],", ",result[2][1, 0],", ",result[2][2, 0],"]")
    print("Count = ",result[1],"\n")