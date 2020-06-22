import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

v = np.array([1,2])
u = np.array([2,1])

def Proj(v, u):
    return (v.dot(u)/(np.linalg.norm(u)**2))*u

def FeatScaling(X):
    Y = X
    Z = np.zeros(Y.shape)
    for i in range(Y.shape[1]):
        max_ = np.max(Y[:,i])
        min_ = np.min(Y[:,i])
        mean_ = np.mean(Y[:,i])
        for j in range(Y.shape[0]):
            if max_ != min_:
                Z[j][i] = (Y[j][i] - mean_)/(max_ - min_)
            else:
                Z[j][i] = Y[j][i] - mean_
    return Z

def PCA(X, dim):
    k = dim
    
    m = X.shape[1]
    X = FeatScaling(X)
    Sigma = (1/m)*X.T.dot(X)
    U = np.linalg.svd(Sigma)[0]
    Ured = U[:, :k]

    return [X.dot(Ured), Ured]

def iter_PCA(X, dim, k0 = 1, threshold = 0.01):
    
    k = k0
    m = X.shape[1]   
    X = FeatScaling(X)
    
    Sigma = (1/m)*X.T.dot(X)
    [U, S, V] = np.linalg.svd(Sigma)
    Ured = U[:, :k]
    SSum = np.sum(S)

    while (1 - (np.sum(S[:k]))/(SSum)) >= threshold:
        k += 1
        Ured = U[:, :k]

    return [X.dot(Ured), Ured, 1 - (np.sum(S[:k]))/(SSum)]

def Reconstruct(Ured, z):
    X = []
    for i in range(z.shape[0]):
        X.append(Ured.dot(z[i]))
    return np.asarray(X)

def PCs(X, Xapprox):
    res = np.sum(np.linalg.norm(X - Xapprox)**2)/np.sum(np.linalg.norm(X)**2)
    print()
    print("Data retained: %f%%" %((1-res)*100))
    return res


X = np.array([
    [1,2,3,4],
    [3,4,5,7],
    [4,2,5,1],
    [9,100,20,30],
    [4,129,30,43],
    [0,392,0,323],
    [21,1,7,3],
    [32,90,19,10]
    ])

print(X1 := FeatScaling(X))

z = PCA(X, 3)
print()

print(z1 := Reconstruct(z[1],z[0]))

plt.figure(0)
plt.imshow(X)
plt.colorbar()

plt.figure(1)
plt.imshow(X1)
plt.colorbar()

plt.figure(2)
plt.imshow(z1)
plt.colorbar()
plt.show()

PCs(X1, z1)

print()
tmp = np.asarray(iter_PCA(X, 4))
print(tmp[0])
print(tmp[1])
print(tmp[2])






