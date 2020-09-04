# !/usr/bin/env python
# Autor: Erik Davino Vincent

import numpy as np
import matplotlib.pyplot as plt

def integral(f, a, b, n):
    return ((b-a)/n) * (f(a)/2 + np.sum(f(np.array([a + k*((b-a)/n) for k in range(1, n)]))) + f(b)/2)

def f(x):
    return (x**2)*(x >= -1)*(x <= 1)

def ao(int_prec, f, L, a, b):
    return (1/L) * integral(f, a, b, int_prec)

def an(n, int_prec, f, L, a, b):

    def g(x):
        return f(x)*np.cos(n*np.pi*x/L)
    
    return (1/L) * integral(g, a, b, int_prec)

def bn(n, int_prec, f, L, a, b):
    
    def g(x):
        return f(x)*np.sin(n*np.pi*x/L)
    
    return (1/L) * integral(g, a, b, int_prec)
    
def fourier_approx(x, func, L, a, b, n, numerical_integral = True, int_prec = 10000):

    if numerical_integral:
        f_sum = ao(int_prec, f, L, a, b)/2
        for m in range(1, n):
            f_sum += an(m, int_prec, func, L, a, b)*np.cos(m*np.pi*x/L) + bn(m, int_prec, func, L, a, b)*np.sin(m*np.pi*x/L)
    else:
        pass
    
    return f_sum

L = 2 # scaler of distance
c = -1 # 100% plot on the negative direction 
d = 1 # 100% plot to the positive direction

t = np.arange(c*L, d*L, .01)
plt.plot(t, fourier_approx(t, f, L, c*L, d*L, 100))
plt.show()


