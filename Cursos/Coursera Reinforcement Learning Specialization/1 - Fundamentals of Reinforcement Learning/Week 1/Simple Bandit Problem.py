# !/usr/bin/env python
# Author: Erik Davino Vincent

import numpy as np
import matplotlib.pyplot as plt

def main():

    k = 4
    eps = 0.5

    Q = np.zeros(k)
    N = np.zeros(k)
    ground_truth = np.random.randn(k)

    plot_cache = []

    while True:

        plot_cache.append(np.mean(Q))

        plt.clf()
        plt.hlines(np.mean(ground_truth), 0, len(plot_cache))
        plt.plot(plot_cache)
        plt.pause(0.01)
        
        Bandit = np.array([np.random.normal(ground_truth[i]) for i in range(k)])

        if np.random.random() < 1 - eps:
            A = np.argmax(Q)
        else:
            A = np.random.randint(0, k)
            
        R = Bandit[A]
        N[A] = N[A] + 1
        Q[A] = Q[A] + (1/N[A])*(R - Q[A])

        

main()
        
