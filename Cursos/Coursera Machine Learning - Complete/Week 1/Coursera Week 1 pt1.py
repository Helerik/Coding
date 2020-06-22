'''
Supervised learning:

Give dataset to algorithm, with known right answers and let the algorithm predict which should be the right answer
for a unknown value (like an interpolation). Regression problem*.

* implies the need to find a continuous solution to your problem.

Classification Problem: Objective is to find a discrete solution to your problem (for example, finding the correct algarism, from 0 to 9).
The decision is given as a probability.

Infinite number of features: (features being the parameters for the learning algorithm.)

---------------------------------------------------------------------------------------------------------------------------------------------

Unsupervised Learning:

Given a dataset, which we dont know the right answer to the problem, we want the learning algorithm to determine what is, if it exists,
the correlation between the data. (Clustering algorithms *)

---------------------------------------------------------------------------------------------------------------------------------------------

Gradient Descent:

Given a supervised learning algorithm, and a cost function for our problem, we use gradient descent to find the minimum (or a minumum) of the
cost function (J(01, 02, 03, ..., 0n))

Have: J(0o, 01)
Want: min {J(0o, 01)}

Start with some initial guess. (0o = O and 01 = O)
Change the parameters and eventually find a minimum.

Inspiration idea: a ball going downhill, as fast as possible --> gradient shows that direction.

Algorithm idea:

repeat until convergence:
    theta_j := theta_j - alpha * d/dtheta_j (J(theta_0, theta_1)), j = 0 and j = 1

alpha is known as learning rate, and indicates the speed of gradient descent
alpha does not need to be changed, for the slope decreases over each step, so the magnitude of each update decreases anyways.

---------------------------------------------------------------------------------------------------------------------------------------------
'''

import math

# Linear regression gradient descent:

theta0 = [0, 0]
alpha = 0.1
points = [(0,1), (2,1), (3, 4), (5,3), (0,2),(10,9)]

def grad_desc(alpha, theta0, points):
    
    m = len(points)
    soma0 = 0
    soma1 = 0
    for i in range(m):
        soma0 += h(points[i][0], theta0) - points[i][1]
        soma1 += (h(points[i][0], theta0) - points[i][1]) * points[i][0]
    
    theta = [theta0[0] - alpha * (1/m) * soma0, theta0[1] - alpha * (1/m) * soma1]

    while True:
        
        soma0 = 0
        soma1 = 0
        for i in range(m):
            soma0 += h(points[i][0], theta) - points[i][1]
            soma1 += (h(points[i][0], theta) - points[i][1]) * points[i][0]

        theta_tmp = [theta[0], theta[1]] 
        theta = [round(theta_tmp[0] - alpha * (1/m) * soma0, 10), round(theta_tmp[1] - alpha * (1/m) * soma1, 10)]
        theta_tmp = [round(theta_tmp[0], 10), round(theta_tmp[1], 10)]

        if math.isnan(theta[0]):
            return "Diverge; escolher outro alpha"

        if [round(theta[0],9), round(theta[1], 9)] == [round(theta_tmp[0],9), round(theta_tmp[1], 9)]:
            return [round(theta[0],9), round(theta[1], 9)]
##        print(theta)

def h(x, theta):
    return theta[0] + theta[1] * x
    

print(grad_desc(alpha, theta0, points))

'''
Above algorithm is known as "batch gradient descent", for it uses all of the training set.















