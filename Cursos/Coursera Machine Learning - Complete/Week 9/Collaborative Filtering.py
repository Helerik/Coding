# Building a recommender system, like the one in netflix, amazon, etc...


import numpy as np
import matplotlib.pyplot as plt

# We could have two optimization objectives, but it is better to have only one!

def CostFunc(X, y, Theta, lamb):
    sum1 = 0
    sum2 = 0
    sum3 = 0

    r = rIndex(y)

    nm = X.shape[0]
    nu = Theta.shape[0]
    n = X.shape[1]

    for i in range(nm):
        for j in range(nu):
            if r[i][j] == 1:
                sum1 += np.square(Theta[j].T.dot(X[i]) - y[i][j])

            for k in range(n):
                sum2 += np.square(X[i][k])
                sum3 += np.square(Theta[j][k])

    sum1 += lamb * (sum2 + sum3)
    sum1 /= 2

    return sum1

def colabFiltering(X, y, Theta, lamb, alpha, iters):
    nm = X.shape[0]
    nu = Theta.shape[0]
    n = X.shape[1]

    r = rIndex(y)

    sumX = 0
    X_tmp = X
    Theta_tmp = Theta

    yind = []
    xind = []
    for i in range(iters):
        xind.append(CostFunc(X, y, Theta, lamb))
        yind.append(i)
        plt.plot(yind,xind, color = 'r')
        plt.pause(0.01)
        for k in range(n):
            for i in range(nm):
                for j in range(nu):
                    if r[i][j] == 1:
                        X_tmp[i][k] -= alpha * ((Theta[j].T.dot(X[i]) - y[i][j]) * Theta[j][k] + lamb*X[i][k])
                        Theta_tmp[j][k] -= alpha * ((Theta[j].T.dot(X[i]) - y[i][j]) * X[i][k] + lamb*Theta[j][k])
                        
        X = X_tmp
        Theta = Theta_tmp

    return [X, Theta]

    
def rIndex(y):
    r = np.zeros(y.shape)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i][j] >= 0:
                r[i][j] = 1

    return r

def findSimilar(movieIndex, X, num):
    i = movieIndex
    
    Normlis = []
    indLis = []
    for j in range(num):
        if j != i:
            Normlis.append(np.linalg.norm(X[i] - X[j]))
            indLis.append(j)
        else:
            Normlis.append(np.linalg.norm(X[i] - X[num]))
            indLis.append(num)

    for j in range(X.shape[0]):
        for k in range(num):
            if (tmp := np.linalg.norm(X[i] - X[j])) < Normlis[k] and j != i and not tmp in Normlis:
                Normlis.pop(ind := Normlis.index(min(Normlis)))
                Normlis.append(tmp)
                indLis.pop(ind)
                indLis.append(j)
                
                break

    return [Normlis, indLis]
        
            

    


# y is the score a person gave a certain movie. If -1, it means the movie hasnt been categorized.
y = np.array([
    [-1,4,0,0,-1,-1,2,3],
    [5,5,-1,0,4,-1,-1,3],
    [5,4,0,-1,4,-1,4,-1],
    [-1,0,-1,5,0,-1,-1,2],
    [0,-1,5,5,-1,5,3,3],
    [0,0,5,-1,0,-1,2,-1],
    ])

# k = number of categories
k = 2

# X.shape[0] = movie || X.shape[1] = movie category
X = np.random.rand(y.shape[0],k)/10

# Theta.shape[0] = person || Theta.shape[1] = how much person likes movie category
Theta = np.random.rand(y.shape[1],k)/10

print("X = \n",(x := colabFiltering(X, y, Theta, 0, 0.015, 100))[0],"\nTheta = \n",x[1])

print()

print("Previous:\n",y, "\nCurrent:\n", z:=np.around(x[0].dot(x[1].T), 0))

print()

print(findSimilar(0, x[0], 1))

plt.close()
plt.plot(X[:,0], X[:,1], 'bo')
plt.show()

plt.figure(1)
plt.imshow(y)
plt.colorbar()
plt.figure(2)
plt.imshow(z)
plt.colorbar()
plt.show()










