from numpy import *

import numpy as np
import scipy.linalg

surfaceMatrix = np.tile(0.0, (1401, 1401))

y = -0.7
x = -0.7
for i in range(0, 1401):
    for j in range(0, 1401):
        surfaceMatrix[i, j] = sqrt(1-((x+(i/1000))**2)-((y+(j/1000))**2))

S, D, VT = np.linalg.svd(surfaceMatrix, full_matrices = True)

Ar = np.zeros((len(S), len(VT)))
Ar += D[0] * np.outer(S.T[0], VT[0])
Ar += D[1] * np.outer(S.T[1], VT[1])

print(Ar)
print("Norm: ",np.linalg.norm(surfaceMatrix-Ar))