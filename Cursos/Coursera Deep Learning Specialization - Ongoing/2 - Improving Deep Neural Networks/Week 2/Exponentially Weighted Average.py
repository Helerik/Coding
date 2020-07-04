# !/usr/bin/env python3

# Implements Exponentially Weighted Average algorithm

import numpy as np
import matplotlib.pyplot as plt

def EWA(data, beta):

    avg = []
    v_i_prev = 0
    
    for i in range(len(data)):

        bias_correction = 1-beta**(i+1)
        
        v_i = beta*v_i_prev + (1-beta)*data[i]
        v_i_prev = v_i
        v_i /= bias_correction
        avg.append(v_i)
        

    return avg

def example():
    X = [9,7,6,4,6,7,6,5,3,4,5,6,1,2,3,1,0,0,3,2,1,4,5,6,7,8,9,9,9,7,8,9,7,6,5,4,5,3,2,2]
    y = range(len(X))
    avg = EWA(X, 0.6)

    plt.scatter(y, X, color = 'r')
    plt.plot(y, avg)

    plt.show()
