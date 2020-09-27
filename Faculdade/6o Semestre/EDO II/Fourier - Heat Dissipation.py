# !/usr/bin/env python
# Autor: Erik Davino Vincent

import numpy as np
import matplotlib.pyplot as plt

##def cn(n):
##    k = n*np.pi
##    return (4*L/k**2)*np.sin(k/2)

def cn(n):
    k = n*np.pi
    return -(L/k)*(np.cos(3*k/4) - np.cos(k/4))
    
def fourier_approx(L, cn, n, x, t):

    f_sum = 0
    for m in range(1, n):
        f_sum += cn(m)*np.sin(m*np.pi*x/L)*np.exp(-t*(m*np.pi/L)**2)
    
    return f_sum

L = 40
n = 10

dx = 0.01
dt = 1

x = np.arange(0, L, dx)
T = int(1500/dt)
sim_speed = 1000

y_max = 0

for t in range(T+1):

    plt.clf()
    
    axes = plt.gca()
    axes.set_xlim([0,L])
    axes.set_ylim([0,3*L/4])
    
    y = fourier_approx(L, cn, n, x, t*dt)
    
    if np.max(y) > y_max:
        y_max = np.max(y)
    c = np.max(y)/y_max
    
    plt.plot(x, y, color = (c, -(2*c-1)**2+1, 1-c))
    plt.title(f"Time elapsed: {t*dt:.3f} s")
    plt.ylabel("Heat")
    plt.xlabel("Bar length")
    plt.pause(dt/sim_speed)



