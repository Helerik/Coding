# !/usr/bin/env python
# Autor: Erik Davino Vincent

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x*(0 < x)*(x < np.pi) + np.pi*(np.pi <= x)*(x < 2*np.pi)

def ao():
    return L/4 + L/2

def an(n):
    return 2 * ( (L/((n*np.pi)**2)) * ( (n*np.pi/2)*np.sin(n*np.pi/2) + np.cos(n*np.pi/2) - 1) - (L/(n*np.pi))*np.sin(n*np.pi/2)/2 )

def bn(n):
    return 0

    
def fourier_approx(L, ao, an, bn, n, x):

    f_sum = ao()/2
    for m in range(1, n):
        f_sum += an(m)*np.cos(m*np.pi*x/L) + bn(m)*np.sin(m*np.pi*x/L)
    
    return f_sum

L = 2*np.pi
n = 100

t = np.arange(-L, L, 0.05)
plt.plot(t, fourier_approx(L, ao, an, bn, n, t))
plt.plot(t, f(t))
plt.show()



