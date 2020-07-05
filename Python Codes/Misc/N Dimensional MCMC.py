# !/usr/bin/env python3
# Author: Erik Davino Vincent

import numpy as np
import matplotlib.pyplot as plt

# g: R^n -> R function
def g(X):
    return -X[0]**4 - X[1]**4 + 1

# N-dimensional normal kernel
def n_dim_kernel(Mu, E):
    
    kernel = [np.random.normal(loc = Mu[i], scale = E[i]) for i in range(len(Mu))]
    
    return kernel

# N-dimensional Monte Carlo Markov Chain algorithm
def MCMC_n_dim(sample_size, E, init):

    dist = [0 for _ in range(sample_size)]
    dist[0] = init
    for i in range(sample_size-1):
        X_nxt = n_dim_kernel(dist[i], E)
        if np.random.uniform(0,1) <= np.minimum(1, g(X_nxt)/g(dist[i])):
            dist[i+1] = X_nxt
        else:
            dist[i+1] = np.copy(dist[i])

    return np.array(dist).T

# Plots the distribution
dist = MCMC_n_dim(500000, E = [0.5,0.5], init = [0.1,0.1])
plt.hist2d(x = dist[0], y = dist[1], bins = 50, density = True)
plt.show()
