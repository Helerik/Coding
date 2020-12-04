# coding: utf-8
# !/usr/bin/env python3
# Author: Erik Davino Vincent
# NUSP: 10736584
# MAP 2320 - Metodos Numericos em Equacoes Diferenciais II

import numpy as np
from numpy import cos, sin, pi, exp
import matplotlib.pyplot as plt
import time

# Solucao manufaturada para ut = uxx + g(x)
def U(x,t):
    return exp(-t)*cos(x) + x*sin(x)

# Derivada de u(x,t) em relacao a x
def Ux(x,t):
    return -exp(-t)*sin(x) + sin(x) + x*cos(x)

# Termo forcante
def g(x):
    return -2*cos(x) + x*sin(x)

# Condicao inicial
def f(x):
    return U(x,0)

# Metodo de Crank-Nicolson para obter solucao numerica do problema com
# condicoes de contorno.
'''
L: comprimento do intervalo de espaco
m: numero de pontos de discretizacao espacial
T: comprimento do intervalo de tempo
k: numero de pontos de discretizacao temporal
alpha: constante da equacao diferencial
do_plot: auxiliar; plota o grafico da solucao em cada instante de tempo se for igual a True
plot_step: a cada quantas iteracoes plota o grafico
'''
def crank_nicolson(L, m, T, k, alpha, do_plot = 1, plot_step = 1):
    
    # h = dx and k = dt
    h = L/m 
    lamb = k*(alpha/h)**2
    jmax = int(T/k)

    # Discretizacao do espaco
    x = np.array([i*h for i in range(m)])
    # Condicao inicial constante igual a 1
    w = f(x).tolist()
    w.append(0)

    l = [1 + lamb]
    u = [-0.5*lamb/l[0]]

    for i in range(1, m-1):
        l.append(1 + lamb + lamb*u[i-1]*0.5)
        u.append(-0.5*lamb/l[i])
    l.append(1 + lamb + lamb*u[m-2]*0.5)

    for j in range(jmax):

        t = j*k
        # Condicao de contorno de Dirichlet
        w[0] = U(0,t)
        # Condicao de contorno de Neumann
        w[m-1] = (2*h*Ux(L,t) + 4*w[m-2] - w[m-3])/3
        
        z = [(w[0] + 0.5*lamb*w[1])/l[0]]
        for i in range(1,m):
            z.append(( (1-lamb)*w[i] + 0.5*lamb*(w[i+1] + w[i-1] + z[i-1]) + k*g(x[i]))/l[i])
        
        for i in range(m-2,-1,-1):
            w[i] = z[i] - u[i]*w[i+1]
        

        if do_plot and j % plot_step == 0:
            plt.clf()
            plt.ylim(0,np.max(w)*1.1)
            plt.plot(x, w[:m])
            plt.plot(x, U(x,t))
            plt.title(f"Tempo elapsado: {j*k:.2}\nPasso atual: {j+1}\nErro: {np.linalg.norm(np.array(w[:m]) - np.array(U(x,t))):.2}")
            plt.pause(0.0001)

def crank_nicolson_fast(L, m, T, k, alpha, do_plot = 1, plot_step = 1):
    
    # h = dx and k = dt
    h = L/m 
    lamb = k*(alpha/h)**2
    jmax = int(T/k)

    # Discretizacao do espaco
    x = np.array([i*h for i in range(m)])
    # Condicao inicial
    w = f(x).tolist()
    w.append(0)

    l = [1 + lamb]
    u = [-0.5*lamb/l[0]]

    for i in range(1, m-1):
        l.append(1 + lamb + lamb*u[i-1]*0.5)
        u.append(-0.5*lamb/l[i])
    l.append(1 + lamb + lamb*u[m-2]*0.5)

    for j in range(jmax):

        t = j*k

        w[0] = U(0,t)
        w[m-1] = (2*h*Ux(L,t) + 4*w[m-2] - w[m-3])/3
        z = [(w[0] + 0.5*lamb*w[1])/l[0]]
        for i in range(1,m):
            z.append(( (1-lamb)*w[i] + 0.5*lamb*(w[i+1] + w[i-1] + z[i-1]) + k*g(x[i]))/l[i])
        
        for i in range(m-2,-1,-1):
            w[i] = z[i] - u[i]*w[i+1]

    return (w[:m], np.linalg.norm(np.array(w[:m]) - np.array(U(x,t))))
        


def main():

    # Visualizacao do Crank-Nicolson
    crank_nicolson(
                L = 1,
                m = 25,
                T = 0.7,
                k = 0.0007,
                alpha = 1,
                do_plot = 1,
                plot_step = 5
                )

main()



