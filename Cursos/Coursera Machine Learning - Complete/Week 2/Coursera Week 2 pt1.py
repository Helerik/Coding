'''
Multilinear linear regression:

Linear regression, but in higher dimensions, i.e. moe features are being analysed

'''

import numpy as np
import math
import matplotlib.pyplot as plt

# hypothesis(x) = theta0 + theta1.x1 + theta2.x2 + ... + thetan.xn

# x0 = 1 by definition. The other x's can be anything else.

def hyp(X, Theta):
    
    if not X[0] == 1:
        return "X[0] must be 1."
    if len(Theta) != len(X):
        return "Length of Theta must be equal to length of X."

    SUM = 0
    for i in range(len(X)):
        SUM += X[i]*Theta[i]
    return SUM

def J(Theta,X,Y):
    n = len(Theta)
    m = len(X)
    SUMS = [0 for i in range(n)]
    for i in range(n):
        for j in range(m):
            SUMS[i] += (hyp(X[j], Theta) - Y[j])**2
    return min(SUMS)/(2*m)

# Multivariate gradient descent:

def gradient_descent(Theta_0, alpha, X, Y, graphics = False):
    n = len(Theta_0)
    Theta = []
    m = len(X)
    SUMS = [0 for i in range(n)]
    for i in range(n):
        for j in range(m):
            SUMS[i] += (hyp(X[j], Theta_0) - Y[j]) * X[j][i]
        Theta.append(Theta_0[i] - (alpha/m) * SUMS[i])
    if graphics == True:
        Jpoints = []
        kpoints = []
        zero = []
        k = -1
    
    while True:
        if graphics == True:
            k += 1
            if k%5 == 0:
                zero.append(0)
                kpoints.append(k)
                Jpoints.append(J(Theta, X, Y))
                plt.plot(kpoints,Jpoints, 'r-', lw = 1)
                plt.plot(kpoints, zero, 'b-', lw = 1)
                plt.pause(0.0001)
        
        Theta_tmp = []
        SUMS = [0 for i in range(n)]
        for i in range(n):
            for j in range(m):
                SUMS[i] += (hyp(X[j], Theta) - Y[j]) * X[j][i]
            Theta_tmp.append(Theta[i])
            Theta[i] = round(Theta_tmp[i] - (alpha/m) * SUMS[i], 10)
            Theta_tmp[i] = round(Theta_tmp[i], 10)
        if math.isnan(Theta[0]):
            return "Diverge; escolher outro alpha."

        if Theta == Theta_tmp:
            return Theta

def Normalize(X):
    for i in range(1, len(X[0])):
        P = [l[i] for l in X]
        maximum = max(P)
        minimum = min(P)
        if minimum == maximum:
            minimum = 0
        summ = 0
        for k in range(len(X)):
            summ += P[k]
        for j in range(len(X)):
            X[j][i] = (X[j][i] - summ/len(X))/(maximum - minimum)

    return X
    
X = [[1,-3],[1,2],[1,3],[1,10]]
Y = [2,3,7,2]
Theta_0 = [0,0]
alpha = 0.01

print(X)
print()
##print(Normalize(X))
##print()
print(gradient_descent(Theta_0, alpha, X, Y))

# Feature Scaling:
# The objective is to make sure features are on a similar scale, so that gradient descent can converge much faster

#good choices for alpha:
# ... 0.001 -> 0.003 -> 0.01 -> 0.03 -> 0.1 -> ...

# Choosing features:
# example:
# Housing prices -> x1 = frontage, x2 = depth
# alternatively: -> x1 = area = frontage * deppth, might be better for the model.

# Polynomial regression:
# With one feature:
#
# hyp(x) = t0 + t1.x1 + t2.x2 + t3.x3 = t0 + t1.x + t2.x^2 + t3.x^3

# Normal Equation:
# Matrix gradient descent:

X = np.matrix([[1,-3,3],\
               [1,2,3],\
               [1,3,4]])
Y = np.array([2,3,7])

Theta = np.linalg.pinv(np.transpose(X).dot(X)).dot(np.transpose(X)).dot(Y)
print()
print(Theta)

# \(OoO)/

# Advantages and disadvantages:
# Gradient Descent: -needs to choose alpha              Normal Equation: +no need to choose alpha
#                   -needs many iterations                               +no need for iteration
#                   +works well for many features                        -needs to compute BIG nxn matrix [O(n^3)]
#                   +good for large n                                    -slow if n is very large (n >= 10000)
#                                                                        -wont work for other models (strictly linear)
#                                                                        -might be non-invertible (+can use pseudo-inverse)

# Causes for non invertibility: n>>m (i.e. too many features); redundant features (i.e. Linearly dependent lines/columns in X)













        
