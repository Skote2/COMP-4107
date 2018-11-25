from numpy import *
from scipy.linalg import *

import numpy as np
import scipy.linalg

A = matrix([[3, 2, -1, 4],
            [1, 0, 2, 3],
            [-2, -2, 3, -1]])

nullA = null_space(A)

print("Rank of the null space matrix is: ", ndim(nullA), " with ", np.size(nullA, 1), " columns")

print("Two linearly independent vectors are:\n")
print(nullA[:, 0], "\n")
print(nullA[:, 1], "\n")

print("Pseudo-Inverse for the matrix is:\n", np.linalg.pinv(A))