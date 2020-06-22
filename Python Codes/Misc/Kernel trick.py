
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




X = np.array([
    [5,5],
    [5,6],
    [5.5,6],
    [5.5,4],
    [4,5.5],
    [4.5,5],
    [4.5,6],
    [4,6],
    [5,6.5],
    [5,10],
    [6,10],
    [5,9],
    [7,8],
    [8,7],
    [9,5],
    [9,6],
    [8,6],
    [8,3],
    [8,4],
    [7,1],
    [6,0],
    [6,1],
    [5,-0.5],
    [5,-1],
    [4,2],
    [4,1],
    [3,3],
    [1,4],
    [1,5],
    [0,5]
    ])

plt.plot(X[:,0], X[:,1], 'bo')
plt.show()

def sim_func(X, X_, sigma):
    return np.exp( - np.square(np.linalg.norm( X - X_)) / (2*np.square(sigma)))

def gauss_map(X, sigma):
    F = np.zeros((X.shape[0], 3))
    for _ in range(X.shape[0]):
        for i in range(X.shape[0]):
            F[_,2] += sim_func(X[_], X[i], sigma)
    F[:,0:2] = X
    return F

X = gauss_map(X, 0.5)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2])
plt.show()
