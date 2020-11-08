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
'''
def forward_difference(L, m, T, k, alpha, do_plot = 1, plot_step = 1):

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
    # Condicao inicial constante igual a 1
    w = [f(x).tolist()]

    for j in range(jmax):

        w.append([(1-2*lamb)*w[j][i] + lamb*(w[j][i-1] + w[j][i+1]) + k*g(x[i]) for i in range(1, m)])

        t = j*k
        # Condicao de contorno de Dirichlet
        w[j+1].insert(0, u(0,t))
        #Condicao de contorno de Neumann
        w[j+1].append((2*h*ux(L,t) + 4*w[j+1][m-1] - w[j+1][m-2])/3)

        if do_plot == 1 and j % plot_step == 0:
            plt.clf()
            plt.ylim(0,np.max(w[-1])*1.1)
            plt.plot(x, w[j])
            plt.plot(x, u(x,j*k))
            plt.title(f"Tempo elapsado: {j*k:.2}\nPasso atual: {j+1}\nErro: {np.linalg.norm(np.array(w[-1]) - np.array(u(x,j*k))):.2}")
            plt.pause(0.0001)

    plt.clf()
    plt.ylim(0,np.max(w[-1])*1.1)
    plt.plot(x, w[-1])
    plt.plot(x, u(x,0.7))
    plt.title(f"Tempo: {T:.2}\nPasso atual: {int(T/k)}\nErro: {np.linalg.norm(np.array(w[-1]) - np.array(u(x,T))):.2}")
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
    # Condicao inicial constante igual a 1
    w = [f(x).tolist()]

    for j in range(jmax):

        w.append([(1-2*lamb)*w[j][i] + lamb*(w[j][i-1] + w[j][i+1]) + k*g(x[i]) for i in range(1, m)])

        t = j*k
        # Condicao de contorno de Dirichlet
        w[j+1].insert(0, u(0,t))
        #Condicao de contorno de Neumann
        w[j+1].append((2*h*ux(L,t) + 4*w[j+1][m-1] - w[j+1][m-2])/3)

    return (w[-1], np.linalg.norm(np.array(w[-1]) - np.array(u(x,T))))

def main():

##    L = 1
##    N = 50
##
##    err_plot = []
##    time_plot = []
##
##    # Avaliacao de tempo de computacao e erro com T variando, m dependente e k dependente
##    T_list = [0.5, 1, 1.5, 2, 2.5, 3]
##    
##    for T in T_list:
##        print(f"Avaliando {N} rodadas; variando T = {T}, fixo lambda ~ 0.49 e 1000 iteracoes")
##
##        k = T/1000
##        m = int(np.sqrt(0.49/k))
##        print("m =", m)
##        print("k =", k)
##
##        timer = time.time()
##        for _ in range(N):
##            err = forward_difference_fast(
##                L = L,
##                m = m,
##                T = T,
##                k = k,
##                alpha = 1
##                )[1]
##        time_plot.append((time.time()-timer)/(N*m))
##        err_plot.append(err)
##        print(f"Tempo medio/m: {time_plot[-1]:.5} s\nErro: {err_plot[-1]:.5}\n"+
##              "================================================================================\n")
##
##    plt.clf()
##    
##    plt.plot(T_list, time_plot)
##    plt.xlabel("T (instante final)")
##    plt.ylabel(f"tempo/({N}*m)")
##    plt.show()
##    
##    plt.plot(T_list, err_plot)
##    plt.xlabel("T (instante final)")
##    plt.ylabel("Erro (||.||2)")
##    plt.show()
##
##    err_plot = []
##    time_plot = []
##
##    # Avaliacao de tempo de computacao e erro com m variando, iter dependente e k dependente
##    m_list = [10, 20, 30, 40, 50]
##    T = 1
##    
##    for m in m_list:
##        print(f"Avaliando {N} rodadas; variando m = {m}, fixo T = {T} e lambda ~ 0.49")
##
##        k = 0.49/(m*m)
##        J = int(T/k)
##        print("iter =", J)
##        print("k =", k)    
##
##        timer = time.time()
##        for _ in range(N):
##            err = forward_difference_fast(
##                L = L,
##                m = m,
##                T = T,
##                k = k,
##                alpha = 1
##                )[1]
##        time_plot.append((time.time()-timer)/(N*J))
##        err_plot.append(err)
##        print(f"Tempo medio/iter: {time_plot[-1]:.5} s\nErro: {err_plot[-1]:.5}\n"+
##              "================================================================================\n")
##        
##    plt.clf()
##    
##    plt.plot(m_list, time_plot)
##    plt.xlabel("m (divisao espacial)")
##    plt.ylabel(f"tempo/({N}*iter)")
##    plt.show()
##    
##    plt.plot(m_list, err_plot)
##    plt.xlabel("m (divisao espacial)")
##    plt.ylabel("Erro (||.||2)")
##    plt.show()

    # Visualizacao do forward difference
    forward_difference(
                L = 1,
                m = 25,
                T = 0.7,
                k = 0.0007,
                alpha = 1,
                do_plot = 1,
                plot_step = 5
                )
main()
    

        








        
    
