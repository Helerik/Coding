# Questao 3 da Tarefa 1.

# Erik Davino Vincent.
# NUSP: 10736584.

import time as tim
import numpy as np
from math import pi, e
import matplotlib.pyplot as plt

def y(t):
    return np.exp(t)*np.sin(2*t)

# derivada de y.
def dy(t, y):
    return np.exp(t)*(np.sin(2*t)+2*np.cos(2*t))

# objetivo: encontrar o valor de y(1), utilizando metodos de passo unico.

# definimos um intervalo de tempo: [t0,tf] = [0,1].
# definimos um passo delta T, ou h: h = (tf-t0)/n, n o numero de subintervalos.
# logo temos: ti = t0 + i*h, i = 1, 2, ..., n

# para esse caso utilizaremos dy como phi.
def euler_explicito(t_0, t_f, passos, func_0, phi):
    timer = tim.time()
    vec = [func_0]
    h = (t_f - t_0)/passos
    t = t_0 + h
    vect = [t]
    
    for k in range(1, passos):
        vec.append(vec[k-1] + h*phi(t, vec[k-1]))
        t = t + h
        vect.append(t)
##    print(vec[-1])
    print("Tempo elapsado:", round(tim.time() - timer,5))
    return vec, vect

# faz o plot da funcao
def main():
    tf = 1
    t0 = 0
    it = 10
    N = [2, 5, 10, 25, 50, 100]
    plt.figure(figsize=(7,5))
    
    for n in (N):
        Y,t = euler_explicito(t0, tf, n, 1, dy)
        print("Erro para n =", n)
        print("Erro =", round(abs(y(1)+1-Y[-1]),3))
        print()
        plt.plot(t,Y, label = 'Plot para n = %d' %n, lw = 1)
    
##    Y,t = euler_explicito(t0, tf, 100, 1, dy)
##    j = np.arange(0., 1., 0.01)
##    plt.plot(j, y(j)+1, label = "y(t)")
##    plt.plot(t,Y, label = 'Plot para n = 100', lw = 1)

    plt.grid(True)
    plt.xlabel('Eixo t')
    plt.ylabel('Eixo y')
    plt.legend()
    plt.show()
    
##    lis = []
##    for n in range(1, 101):
##        Y,t = euler_explicito(t0, tf, n, 1, dy)
##        lis.append(Y[-1])
##    plt.plot([i for i in range(1,101)],lis)
##    plt.grid(True)
##    plt.xlabel('n')
##    plt.ylabel('y(1)')
##    plt.show()
    
    
main()




