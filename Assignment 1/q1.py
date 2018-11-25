from numpy import *

import numpy as np
import scipy.linalg

ratings = np.matrix([[3, 1, 2, 3],
                     [4, 3, 4, 3],
                     [3, 2, 1, 5],
                     [1, 6, 5, 2]])

Alicia = np.mean(ratings[0, :])

print(ratings,"\n")

print("Alicias average rating: ",Alicia,"\n")

S, D, VT = np.linalg.svd(ratings, full_matrices = True)

S = S[:, 0:2]
D = diag(D)
VT = VT[0:2, :]

print("\n")
print(np.matrix(S))
print("\n")
print(np.matrix(D)[0:2, 0:2])
print("\n")
print(np.matrix(VT))

Alicia += S[3, :].dot(D[0:2, 0:2].dot(VT[:, 0]))

print("\n")
print("Alicias expected rating: ",Alicia)

Alice = matrix([[5], [3], [4], [4]])

print("\n\n\n")
print(Alice)

D = linalg.inv(D[0:2, 0:2])

print("\n")
print(D)
print("\n")
print(VT)
print("\n")
print(D.dot(VT.dot(Alice)))

#print(val)