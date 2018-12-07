#testing that what I'm doing for q1 fits the theory from the slides
import numpy as np

def sgn (num):
    if (num >= 0):
            return 1
    return -1

x = [
    np.array([ 1,-1,-1, 1]),
    np.array([ 1, 1,-1, 1]),
    np.array([-1, 1, 1,-1]),
]
dim = x[0].size
w = np.zeros([dim, dim])
for v in x:
    vt = np.transpose(v)
    w += np.outer(v, v)
    #np.outer(data[1], data[1])
w -= len(x) * np.identity(dim)

y = [
    np.empty(dim, x[0].dtype),
    np.empty(dim, x[0].dtype),
    np.empty(dim, x[0].dtype)
]

for v in range(len(x)):
    for i in range(dim):
        sum = 0
        for j in range(dim):
            if (x[v][j] == 1):
                sum += w[i][j]*1
            #else: sum += w...*0
        y[v][i] = sgn(sum)
print(x)
print(y)