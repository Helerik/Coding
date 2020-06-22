# Density estimation algorithm (for anomaly detection)

import numpy as np
import matplotlib.pyplot as plt

def normal(x, mu, sig2):
    return (1/np.sqrt(2*np.pi*sig2)) * np.exp(-((x - mu)**2)/(2*sig2))

def multiNormal(x, mu, SIG):
    n = len(mu)
    return (1/(np.power(2*np.pi, n/2)*np.sqrt(np.linalg.det(SIG)))) * np.exp((-1/2) * ((x - mu).T.dot(np.linalg.inv(SIG))).dot(x - mu))

def anomalyDetec(Xtrain, Xtest, eps):
    m = Xtrain.shape[0]
    mu = (1/m) * np.sum(Xtrain, axis = 0)
    sig2 = (1/m) * np.sum((Xtrain - mu)**2, axis = 0)

    p = 1
    for i in range(len(Xtest)):
        p *= normal(Xtest[i], mu[i], sig2[i])
    
    if p < eps:
        print("\nAnomaly detected. P(x) = %f" %p)
        return True
    else:
        print("\nNo anomaly detected. P(x) = %f" %p)
        return False

def anomalyLabel(Xtrain, Xtest, eps):
    m = Xtrain.shape[0]
    mu = (1/m) * np.sum(Xtrain, axis = 0)
    sig2 = (1/m) * np.sum((Xtrain - mu)**2, axis = 0)

    labels = np.zeros((Xtest.shape[0], 1))

    for i in range(Xtest.shape[0]):
        p = 1
        for j in range(Xtest.shape[1]):
            p *= normal(Xtest[i][j], mu[j], sig2[j])
        if p < eps:
            labels[i] = 1

    return labels

def anomalyDetec_Mult(Xtrain, Xtest, eps):
    m = Xtrain.shape[0]
    mu = ((1/m) * np.sum(Xtrain, axis = 0)).T
    SIG = (Xtrain - mu).T.dot(Xtrain - mu)
    SIG /= m - 1
    
    p = multiNormal(Xtest, mu, SIG)
    
    if p < eps:
        return True
    else:
        return False

def anomalyLabel_Mult(Xtrain, Xtest, eps):
    m = Xtrain.shape[0]
    mu = ((1/m) * np.sum(Xtrain, axis = 0)).T
    SIG = (Xtrain - mu).T.dot(Xtrain - mu)
    SIG /= m - 1

    labels = np.zeros((Xtest.shape[0], 1))

    for i in range(Xtest.shape[0]):
        if multiNormal(Xtest[i], mu, SIG) < eps:
            labels[i] = 1

    return labels  

Xtrain = np.array([
    [10,10],
    [10,11],
    [11,10],
    [10,9],
    [12,10],
    [9,9],
    [8,8],
    [8,10],
    [9.5,10],
    [12,10],
    [9.8,9],
    [18,10],
    [2,9.3],
    [11,8]
    ])

Xtest = np.array([
    [10,10],
    [12,10],
    [20,1],
    [10,12],
    [8,10]
     ])

print(anomalyLabel(Xtrain, Xtest, 0.01))

plt.plot(Xtrain[:,0], Xtrain[:,1], 'bx')
plt.plot(Xtest[:,0], Xtest[:,1], 'rx')
plt.show()

print(anomalyLabel_Mult(Xtrain, Xtest, 0.01))

x = np.arange(-5, 5, 0.5)
z = np.zeros((len(x), len(x)))

for i in range(len(x)):
    for j in range(len(x)):
        z[i][j] = multiNormal(np.array([x[i], x[j]]), np.array([0,0]), np.array([[2,0],[1,2]]))

plt.imshow(z, interpolation = 'bicubic')
plt.colorbar()
plt.show()













