# coding: utf-8
# !/usr/bin/env python3
# Author: Erik Davino Vincent
# NUSP: 10736584
# MAP 2320 - Metodos Numericos em Equacoes Diferenciais II

'''
Problema:
          Resolver a equacao da onda na corda (1D) para as seguintes condicoes iniciais
          i)   f(x) = {0 se 0 <= x < 1/3;
                      {1 se 1/3 <= x < 2/3;
                      {0 se 2/3 <= x <= 1

          ii)  f(x) = {0 se 0 <= x < 1/4;
                      {4(x-1/4) se 1/4 < x < 1/2;
                      {4(3/4-x) se 1/2 <= x < 3/4;
                      {0 se 3/4 <= x <= 1

          iii) f(x) = exp(-50(x-1/2)^2) 

'''

import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
import time

# Funcoes a serem resolvidas
def fun1(x):
    if 1/3 <= x < 2/3:
        return 1
    else:
        return 0

def fun2(x):
    if 1/4 < x < 1/2:
        return 4*(x-1/4)
    elif 1/2 <= x < 3/4:
        return 4*(3/4-x)
    else:
        return 0

def fun3(x):
    return exp(-50*(x-1/2)**2)

# Extra
def fun4(x):
    return np.sin(2*np.pi*x)

# Contorno
def g(x):
    return 0

# Metodo de diferencas finitas para equacao da onda em 1D (corda)
def diff_finit(m, n, L, T, alpha, fun, g):

    h = L/m
    k = T/n
    lamb = k*alpha/h
##    print(f"\nLambda = {lamb};")
    if lamb > 1:
        print("Lambda > 1; escolha outros parametros.")
        print(lamb)
        return

    
    W = [[0 for i in range(0,m+1)] for j in range(n+1)]
    W[0][0] = fun(0)
    W[0][m] = fun(L)
    for i in range(1, m):
        W[0][i] = fun(i*h)
        W[1][i] = (1-lamb**2)*fun(i*h) + lamb*lamb*(fun((i+1)*h) + fun((i-1)*h))/2 + k*g(i*h)

    for j in range(1, n):
        for i in range(1, m):
            W[j+1][i] = 2*(1-lamb**2)*W[j][i] + lamb*lamb*(W[j][i+1]+ W[j][i-1]) - W[j-1][i]

    return W

def visualize(m,n,L,T,init):
    W = diff_finit(m = m,
                   n = n,
                   L = L,
                   T = T,
                   alpha = 1,
                   fun = init,
                   g = g)
    X = [x*L/m for x in range(0,m+1)]
    for i in range(0, n, 25):
        plt.clf()
        plt.plot(X, W[i])
        plt.title(f'Simulacao Funcao de Onda (1D)\nTempo elapsado: {i*T/n:>5.2f}u')
        plt.ylim(-1.1,1.1)
        plt.pause(0.001)
    plt.clf()
    plt.plot(X, W[-1])
    plt.title(f'Simulacao Funcao de Onda (1D)\nTempo elapsado: {T}u')
    plt.ylim(-1.1,1.1)
    plt.show()

    err = np.linalg.norm(np.array(W[0]) - np.array(W[-1]))
    print(f"Erro Quadratico = {err}\nErro Quadratico Medio = {err/m}")

init = fun1

visualize(m = 125,
          n = 500,
          L = 1,
          T = 2,
          init = init)
visualize(m = 250,
          n = 1000,
          L = 1,
          T = 2,
          init = init)
visualize(m = 250,
          n = 500,
          L = 1,
          T = 2,
          init = init)
visualize(m = 500,
          n = 1000,
          L = 1,
          T = 2,
          init = init)
input('\nPressione enter para continuar:')

# Analise do erro:
L = 1
T = 2
print()
for n in [100, 200, 500, 1000]:
    for m in [n/10, 2*n/10, 3*n/10, 4*n/10, n/2]:
        W = diff_finit(m = int(m),
                   n = n,
                   L = L,
                   T = T,
                   alpha = 1,
                   fun = init,
                   g = g)
        err = np.linalg.norm(np.array(W[0]) - np.array(W[-1]))/m
        print(f"n = {n}; m = {m};\nErro Quadratico = {err}\nErro Quadratico Medio = {err/m}")

# Analise de tempo computacional
L = 1
T = 2
q = 33
for n in [100, 200, 500, 1000]:
    for m in [n/10, 2*n/10, 3*n/10, 4*n/10, n/2]:
        t = time.time()
        for init in [fun1, fun2, fun3]:
            for _ in range(q):
                diff_finit(m = int(m),
                       n = n,
                       L = L,
                       T = T,
                       alpha = 1,
                       fun = init,
                       g = g)
        print(f"\nn = {n}; m = {m};\nTempo elapsado medio = {(time.time() - t)/(q*3)}")

# Analise grafica (corda longa):
init = fun3
visualize(m = 500,
          n = 2000,
          L = 10,
          T = 20,
          init = init)

# Funcao de onda "natural"
init = fun4
visualize(m = 1000,
          n = 500,
          L = 1,
          T = 0.5,
          init = init)









