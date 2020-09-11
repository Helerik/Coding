# !/usr/bin/env python
# Autor: Erik Davino Vincent

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x*(0 < x)*(x < np.pi)

def ao():
    return np.pi/2

def an(n):
    return (-1+(-1)**n)/(np.pi * n**2)

def bn(n):
    return (-(-1)**n)/(n)

    
def fourier_approx(L, ao, an, bn, n, x):

    f_sum = ao()/2
    for m in range(1, n):
        f_sum += an(m)*np.cos(m*np.pi*x/L) + bn(m)*np.sin(m*np.pi*x/L)
    
    return f_sum

L = np.pi
n = 10

t = np.arange(-2*np.pi, 2*np.pi, 0.05)
plt.plot(t, fourier_approx(L, ao, an, bn, n, t))
plt.plot(t, f(t))
plt.show()



