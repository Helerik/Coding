# !/usr/bin/env python3

# Implements Exponentially Weighted Average algorithm

import numpy as np
import matplotlib.pyplot as plt

def EWA(data, beta):

    avg = []
    v_t_prev = 0
    
    for i in range(len(data)):
        
        v_t = beta*v_t_prev + (1-beta)*data[i]
        avg.append(v_t)
        v_t_prev = v_t

    return avg

def example():
    X = [0,1,4,5,6,7,6,5,3,4,5,6,1,2,3,1,0,0,3,2,1,4,5,6,7,8,9,9,9,7,8,9,7,6,5,4,5,3,2,2]
    y = range(len(X))
    avg = EWA(X, 0.5)

    plt.scatter(y, X, color = 'r')
    plt.plot(y, avg)

    plt.show()
