from numpy import *

import numpy as np
import scipy.linalg

inMatrix = matrix([[1, 2, 3],
                   [2, 3, 4],
                   [4, 5, 6],
                   [1, 1, 1]])

S, D, VT = np.linalg.svd(inMatrix, full_matrices = True)

print("Input matrix: ",inMatrix,"\n")
print("S: ", S, "\n")
print("D: ", diag(D), "\n")
print("VT: ", VT)