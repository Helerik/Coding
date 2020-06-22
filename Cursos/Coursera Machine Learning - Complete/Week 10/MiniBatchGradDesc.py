
import numpy as np
import matplotlib.pyplot as plt

def miniBatchGradDesc(X, y, Theta, alpha, b, iters):
    m = X.shape[0]
    n = X.shape[1]
    Theta_last = Theta
    
    for it in range(iters):        
        for i in range(0, m, b):
            y_mini = y[i:i+b]
            X_mini = X[i:i+b,:]
            h = X_mini.dot(Theta.T)

##            plt.plot(Theta[0],Theta[1], 'bx')
##            plt.pause(0.01)

            Theta_last = Theta
            Theta = Theta - alpha * (X_mini.T.dot(h - y_mini))/b
            
            
    return Theta
        
X = np.array([
    [1,2],
    [1,3],
    [1,5],
    [1,6],
    [1,8],
    [1,9],
    [1,10],
    [1,13],
    [1,14],
    [1,15],
    [1,16],
    [1,15],
    [1,18],
    [1,19],
    [1,20],
    [1,21]
    ])

X_tmp = np.zeros((X.shape[0], 3))
X_tmp[:,0:2] = X
X_tmp[:, 2] = X_tmp[:, 1]**2
X = X_tmp

y = np.array([1,2,3,4,6,7,8,9,10,11,13,12,14,15,16,17])

theta = np.array([0,0,0])

print(miniBatchGradDesc(X, y, theta, 0.00001, 2, 100000))



























