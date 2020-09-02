# !/usr/bin/env python
# Author: Erik Davino Vincent

import numpy as np
import matplotlib.pyplot as plt

def main():

    k = 10
    eps1 = 0.1
    eps2 = 0.1
    
    alpha = 0.5
    beta = alpha
    On = 0

    Q1 = np.zeros(k)
    Q2 = np.zeros(k)
    ground_truth = np.random.randn(k) + 10

    plot_cache = [[],[]]

    while True:

        plot_cache[0].append(np.mean(Q1))
        plot_cache[1].append(np.mean(Q2))

        plt.clf()
        plt.hlines(np.mean(ground_truth), 0, len(plot_cache[0]))
        plt.plot(plot_cache[0], label = "alpha")
        plt.plot(plot_cache[1], label = "beta")
        plt.legend()
        plt.pause(0.01)        
        Bandit = np.array([np.random.normal(ground_truth[i]) for i in range(k)])
        ground_truth = np.array([np.random.normal(ground_truth[i]) for i in range(k)])

        if np.random.random() < 1 - eps1:
            A = np.argmax(Q1)
        else:
            A = np.random.randint(0, k)
            
        R = Bandit[A]
        Q1[A] = Q1[A] + alpha*(R - Q1[A])

        if np.random.random() < 1 - eps2:
            A = np.argmax(Q2)
        else:
            A = np.random.randint(0, k)
            
        R = Bandit[A]
        On = On + alpha*(1 - On)
        beta = alpha/On
        Q2[A] = Q2[A] + beta*(R - Q2[A])


main()
        
