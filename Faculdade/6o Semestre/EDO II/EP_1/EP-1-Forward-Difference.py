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

# Metodo de forward difference para obter solucao numerica do problema com
# condicoes de contorno.
'''
L: comprimento do intervalo de espaco
m: numero de pontos de discretizacao espacial
T: comprimento do intervalo de tempo
k: numero de pontos de discretizacao temporal
alpha: constante da equacao diferencial
do_plot: auxiliar; plota o grafico da solucao em cada instante de tempo se for igual a True
plot_step: a cada quantas iteracoes plota o grafico
err: analise o erro
'''
def forward_difference(L, m, T, k, alpha, do_plot = 1, plot_step = 1, err = 1):

    # h = dx and k = dt
    h = L/m 
    lamb = k*(alpha/h)**2
    print("Lambda =", lamb)
    if lamb > 0.5:
        print("Lambda fora do limite de estabilidade por", lamb-0.5)
        return
    jmax = int(T/k)

    # Discretizacao do espaco
    x = np.array([i*h for i in range(m+1)])
    # Condicao inicial
    w = [f(x).tolist()]

    if do_plot:
        plt.clf()
        plt.ylim(0,np.max(w[-1])*1.1)
        plt.xlabel("x")
        plt.ylabel(f"u(x,0)")
        plt.plot(x, w[0])
        plt.title(f"Tempo elapsado: 0\nPasso atual: {0}\nErro: {np.linalg.norm(np.array(w[-1]) - np.array(u(x,0))):.2}")
        plt.show()

    if err:
        E = [0]

    for j in range(jmax):

        w.append([(1-2*lamb)*w[j][i] + lamb*(w[j][i-1] + w[j][i+1]) + k*g(x[i]) for i in range(1, m)])

        t = j*k
        # Condicao de contorno de Dirichlet
        w[j+1].insert(0, u(0,t+k))
        #Condicao de contorno de Neumann
        w[j+1].append((2*h*ux(L,t+k) + 4*w[j+1][m-1] - w[j+1][m-2])/3)

        if do_plot and j % plot_step == 0:
            plt.clf()
            plt.ylim(0,np.max(w[-1])*1.1)
            plt.xlabel("x")
            plt.ylabel(f"u(x,{t:.2})")
            plt.plot(x, w[j])
            plt.title(f"Tempo elapsado: {j*k:.2}\nPasso atual: {j+1}\nErro: {np.linalg.norm(np.array(w[-1]) - np.array(u(x,t+k))):.2}")
            plt.pause(0.0001)

        if err:
            E.append(np.linalg.norm(np.array(w[-1]) - np.array(u(x,t+k))))

    if do_plot:
        plt.clf()
        plt.ylim(0,np.max(w[-1])*1.1)
        plt.xlabel("x")
        plt.ylabel(f"u(x,{j*k:.2})")
        plt.plot(x, w[j])
        plt.title(f"Tempo elapsado: {j*k:.2}\nPasso atual: {j+1}\nErro: {np.linalg.norm(np.array(w[-1]) - np.array(u(x,t+k))):.2}")
        plt.show()

    if err:
        plt.clf()
        plt.xlabel("passo")
        plt.ylabel("erro")
        plt.title("Erro para cada passo")
        plt.plot(E)
        plt.show()

# Metodo de forward difference para obter solucao numerica do problema com
# condicoes de contorno.
'''
L: comprimento do intervalo de espaco
m: numero de pontos de discretizacao espacial
T: comprimento do intervalo de tempo
k: numero de pontos de discretizacao temporal
alpha: constante da equacao diferencial
'''
def forward_difference_fast(L, m, T, k, alpha):
    
    # h = dx and k = dt
    h = L/m 
    lamb = k*(alpha/h)**2
    if lamb > 0.5:
        print("Lambda fora do limite de estabilidade por", lamb-0.5)
        return
    jmax = int(T/k)

    # Discretizacao do espaco
    x = np.array([i*h for i in range(m+1)])
    # Condicao inicial
    w = [f(x).tolist()]

    for j in range(jmax):

        w.append([(1-2*lamb)*w[j][i] + lamb*(w[j][i-1] + w[j][i+1]) + k*g(x[i]) for i in range(1, m)])

        t = j*k
        # Condicao de contorno de Dirichlet
        w[j+1].insert(0, u(0,t+k))
        #Condicao de contorno de Neumann
        w[j+1].append((2*h*ux(L,t+k) + 4*w[j+1][m-1] - w[j+1][m-2])/3)

def main():

    # Visualizacao do forward difference
    forward_difference(
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
        forward_difference_fast(
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
        forward_difference_fast(
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
        forward_difference_fast(
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
        forward_difference_fast(
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
    

        








        
    
