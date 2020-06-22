import numpy as np
import time

def LinRegCost(theta, X, y, lamb = 0):
    h = X.dot(theta.T)
    m = y.size
    return (np.sum(np.power(h - y, 2)) + lamb*np.sum(np.power(theta, 2)))/(2*m)

def LogRegCost(theta, X, y, lamb = 0):
    h = X.dot(theta.T)
    m = y.size
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))/m + lamb*np.sum(np.power(theta, 2))/(2*m)

def GradDescLinReg(theta, X, y, alpha, num_iters, lamb = 0):
    m = y.size
    for k in range (num_iters):
        h = X.dot(theta.T)
        theta = theta - alpha*(X.T.dot(h-y))/m
        if lamb != 0:
            theta[1:] = theta[1:] - alpha*lamb*theta[1:]/m
    return theta

def GradDescLogReg(theta, X, y, alpha, num_iters, lamb = 0):
    m = y.size
    for k in range (num_iters):
        h = 1/(1+np.exp(-X.dot(theta.T)))
        theta = theta - alpha*(X.T.dot(h-y))/m
        if lamb != 0:
            theta[1:] = theta[1:] - alpha*lamb*theta[1:]/m
    return theta

def FeatScaling(X):
    Y = X.T
    Z = np.zeros(Y.shape)
    for i in range(Y.shape[0]):
        max_ = np.max(Y[i])
        min_ = np.min(Y[i])
        mean_ = np.mean(Y[i])
        for j in range(Y.shape[1]):
            if max_ != min_:
                Z[i][j] = (Y[i][j] - mean_)/(max_ - min_)
            else:
                Z[i][j] = Y[i][j] - mean_
    return Z.T


      
X = np.array([
    [1,0,0],
    [1,1,1],
    [1,2,4],
    [1,3,9],
    [1,4,16],
    [1,5,25]
    ])
y = np.array([0.1,1.1,4.5,8,16.7,23])
theta = np.array([2,2,2])
alpha = 0.01
num_iters = 1000
for i in range(1,6):
    print(GradDescLinReg(theta, X, y, alpha, 10**i))

