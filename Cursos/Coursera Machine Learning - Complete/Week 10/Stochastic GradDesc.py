# Another type of gradient descent that is useful for a very large training set

import numpy as np
import matplotlib.pyplot as plt

def LinRegCost(theta, X, y, lamb = 0):
    h = X.dot(theta.T)
    m = y.size
    return (np.sum(np.power(h - y, 2)) + lamb*np.sum(np.power(theta, 2)))/(2*m)

def shuffle(a, b):
    ind = np.asarray([i for i in range(len(b))])
    np.random.shuffle(ind)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)

    for i in range(len(b)):
        shuffled_a[i] = a[ind[i]]
        shuffled_b[i] = b[ind[i]]
        
    return [shuffled_a, shuffled_b]

def stochGradDesc(X, y, Theta, alpha, iters):
    [X,y] = shuffle(X,y)
    m = X.shape[0]
    n = X.shape[1]

    cost = []
    its = []
    
    for it in range(iters):
        # h may be some other function...
        h = X.dot(Theta.T)

##        its.append(it)
##        cost.append(LinRegCost(Theta, X, y))
##        plt.plot(cost, its, 'r')
##        plt.pause(0.01)
        
        for i in range(m):
            Theta = Theta - alpha * (h[i] - y[i])*X[i]
            
##            plt.plot(Theta[0], Theta[1], 'bx')
##            plt.pause(0.01)

            

    return Theta

np.random.seed([10,10])

m = 100

X = np.random.random((m,1))*100

y = np.random.random(m)*100

X_tmp = np.ones((m,2))
X_tmp[:,1:] = X

X = X_tmp

Theta = np.array([1,1])

print(stochGradDesc(X, y, Theta, 0.000005, 100000))











