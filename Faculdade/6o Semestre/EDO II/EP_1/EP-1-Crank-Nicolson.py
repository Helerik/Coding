# coding: utf-8
# !/usr/bin/env python3
# Author: Erik Davino Vincent
# NUSP: 10736584
# MAP 2320 - Metodos Numericos em Equacoes Diferenciais II

import numpy as np
from numpy import cos, sin, pi, exp
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.sparse import diags
import time

# Solucao manufaturada para ut = uxx + g(x)
def u(x,t):
    return exp(-t)*cos(x) + x*sin(x)

# Derivada de u(x,t) em relacao a x
def ux(x,t):
    return -exp(-t)*sin(x) + sin(x) + x*cos(x)

# Termo forcante
def g(x):
    return -2*cos(x) + x*sin(x)

# Condicao inicial
def f(x):
    return u(x,0)

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
err: analisa o erro
'''
def crank_nicolson(L, m, T, k, alpha, do_plot = 1, plot_step = 1, err = 1):
    
    # h = dx and k = dt
    h = L/m 
    lamb = k*(alpha/h)**2
    jmax = int(T/k)

    # Discretizacao do espaco
    x = np.array([i*h for i in range(m+1)])
    # Condicao inicial
    w = f(x).tolist()

    # Matriz do lado esquerdo
    M = diags([-lamb, 2*(1+lamb), -lamb], [-1, 0, 1], shape=(m-1, m-1)).toarray()
    
    # C.C Neumann
    M[-1, -2] = -2*lamb/3
    M[-1, -1] = 2 + 2*lamb/3
    
    N = inv(M)

    if do_plot:
            plt.clf()
            plt.ylim(0,np.max(w)*1.1)
            plt.xlabel("x")
            plt.ylabel(f"u(x,0)")
            plt.plot(x, w)
            plt.title(f"Tempo elapsado: 0\nPasso atual: 0\nErro: {np.linalg.norm(np.array(w) - np.array(u(x,0))):.2}")
            plt.show()

    if err:
        E = [0]

    for j in range(jmax):

        t = j*k
        b = np.array([2*(1-lamb)*w[i] + lamb*(w[i-1] + w[i+1]) for i in range(1, m)])

        # C.C Dirichlet
        b[0] += lamb*u(0,t)
        # C.C Neumann
        b[-1] += (2*lamb*h*ux(L,t+k))/3

        w[0] = u(0,t+k)
        w[1:-1] = np.dot(N,b) + k*g(x[1:-1])
        w[m] = (2*h*ux(L,t+k) + 4*w[m-1] - w[m-2])/3

        if do_plot and j % plot_step == 0:
            plt.clf()
            plt.ylim(0,np.max(w)*1.1)
            plt.xlabel("x")
            plt.ylabel(f"u(x,{t:.2})")
            plt.plot(x, w)
            plt.title(f"Tempo elapsado: {j*k:.2}\nPasso atual: {j+1}\nErro: {np.linalg.norm(np.array(w) - np.array(u(x,t+k))):.2}")
            plt.pause(0.0001)

        if err:
            E.append(np.linalg.norm(np.array(w) - np.array(u(x,t+k))))
            
    if do_plot:
        plt.clf()
        plt.ylim(0,np.max(w)*1.1)
        plt.xlabel("x")
        plt.ylabel(f"u(x,{t:.2})")
        plt.plot(x, w)
        plt.title(f"Tempo elapsado: {j*k:.2}\nPasso atual: {j+1}\nErro: {np.linalg.norm(np.array(w) - np.array(u(x,t+k))):.2}")
        plt.show()

    if err:
        plt.clf()
        plt.xlabel("passo")
        plt.ylabel("erro")
        plt.title("Erro para cada passo")
        plt.plot(E)
        plt.show()

def crank_nicolson_fast(L, m, T, k, alpha):
    
    # h = dx and k = dt
    h = L/m 
    lamb = k*(alpha/h)**2
    jmax = int(T/k)

    # Discretizacao do espaco
    x = np.array([i*h for i in range(m+1)])
    # Condicao inicial
    w = f(x).tolist()

    # Matriz do lado esquerdo
    M = diags([-lamb, 2*(1+lamb), -lamb], [-1, 0, 1], shape=(m-1, m-1)).toarray()
    
    # C.C Neumann
    M[-1, -2] = -2*lamb/3
    M[-1, -1] = 2 + 2*lamb/3

    N = inv(M)

    for j in range(jmax):

        t = j*k
        b = np.array([2*(1-lamb)*w[i] + lamb*(w[i-1] + w[i+1]) for i in range(1, m)])

        # C.C Dirichlet
        b[0] += lamb*u(0,t)
        # C.C Neumann
        b[-1] += (2*lamb*h*ux(L,t+k))/3

        w[0] = u(0,t+k)
        w[1:-1] = np.dot(N,b) + k*g(x[1:-1])
        w[m] = (2*h*ux(L,t+k) + 4*w[m-1] - w[m-2])/3

def main():

    # Visualizacao do Crank-Nicolson
    crank_nicolson(
                L = 1,
                m = 25,
                T = 0.79,
                k = 0.00079,
                alpha = 1,
                do_plot = 1,
                plot_step = 10,
                err = 1
                )

    # Analise do tempo computacional
    m = 10
    k = 0.00079
    timer = time.time()
    for _ in range(100):
        crank_nicolson_fast(
            L = 1,
            m = m,
            T = 0.79,
            k = k,
            alpha = 1
            )
    t = time.time() - timer
    print()
    print(f"m={m}, k={k}\nTempo medio em 100 rodadas: {t/100:.5}")

    m = 10
    k = 0.000079
    timer = time.time()
    for _ in range(100):
        crank_nicolson_fast(
            L = 1,
            m = m,
            T = 0.79,
            k = k,
            alpha = 1
            )
    t = time.time() - timer
    print()
    print(f"m={m}, k={k}\nTempo medio em 100 rodadas: {t/100:.5}")

    m = 25
    k = 0.00079
    timer = time.time()
    for _ in range(100):
        crank_nicolson_fast(
            L = 1,
            m = m,
            T = 0.79,
            k = k,
            alpha = 1
            )
    t = time.time() - timer
    print()
    print(f"m={m}, k={k}\nTempo medio em 100 rodadas: {t/100:.5}")

    m = 25
    k = 0.000079
    timer = time.time()
    for _ in range(100):
        crank_nicolson_fast(
            L = 1,
            m = m,
            T = 0.79,
            k = k,
            alpha = 1
            )
    t = time.time() - timer
    print()
    print(f"m={m}, k={k}\nTempo medio em 100 rodadas: {t/100:.5}")

main()



