# coding: utf-8
# !/usr/bin/env python3
# Author: Erik Davino Vincent
# NUSP: 10736584
# MAP 2320 - Metodos Numericos em Equacoes Diferenciais II

'''
Problema:
          resolver por metodos numericos o problema uxx(x,y) + uyy(x,y) = f(x,y)
          para o problema com solucao manufaturada u(x,y) = cos(pi*x)*cos(pi*y).
Metodos:
          Esquema 1: Wj,i = (Wj,i+1 + Wj,i-1 + Wj+1,i + Wj-1,i + h*h*Fj,i)/4
          Esquema 2: Wj,i = (Wj+1,i+1 + Wj-1,i-1 + Wj+1,i-1 + Wj-1,i+1 + 2*h*h*Fj,i)/4
          Esquema 3: Wj,i = (Wj,i+1 + Wj,i-1 + Wj+1,i + Wj-1,i)/5 +
                            (Wj+1,i+1 + Wj-1,i-1 + Wj+1,i-1 + Wj-1,i+1)/20 +
                             h*h*(Fj,i+1 + Fj,i-1 + Fj+1,i + Fj-1,i + 8*Fi,j)/40
'''

import numpy as np
from numpy import cos, sin, pi, exp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time

# Solucao manufaturada para uxx + uyy = f(x,y)
def u(x,y): return cos(pi*x)*cos(pi*y)
# Derivada de u(x,y) em relacao a x duas vezes
def uxx(x,y): return -pi*pi*u(x,y)
# Derivada de u(x,y) em relacao a y duas vezes
def uyy(x,y): return -pi*pi*u(x,y)
# Termo forcante
f = lambda x, y : uxx(x,y) + uyy(x,y)

'''
m, n : numero de pontos de discretizacao em x e y respectivamente.
X : intervalo no eixo x.
Y : intervalo no eixo y.
max_iter : numero maximo de iteracoes.
tol : se a diferenca entre a norma infinita do passo anterior e do passo atual for <= tol, retorna.
do_plot : se verdadeiro, plota "mapa de calor" da funcao aproximada.
plot_step : a cada quantos passos apresenta o plot se do_plot = True.
do_print : se verdadeiro, printa mensagem ao final da execucao.
'''
def scheme(m, n,
           esquema = 1,
           X = [0,1],
           Y = [0,1],
           max_iter = 1000,
           tol = 1e-5,
           do_plot = True,
           plot_step = 5,
           do_print = True):
    
    if plot_step <= 0:
        do_plot = False

    # Passo de discretizacao
    h = (X[1]-X[0])/m
    k = (Y[1]-Y[0])/n

    # Malha discretizada
    x = np.array([X[0] + i*h for i in range(m+1)])
    y = np.array([Y[0] + j*k for j in range(n+1)])

    # Estado inicial
    W = np.zeros((n+1,m+1))
    F = [[f(x[i],y[j]) for i in range(m+1)] for j in range(n+1)]

    # Condicoes de contorno
    W[0, :] = u(Y[1],x)
    W[-1,:] = u(Y[0],x)
    W[:, 0] = u(y,X[1])
    W[:,-1] = u(y,X[0])

    # Inicializa norma
    norm = np.inf

    for p in range(max_iter):

        W_prev = W.copy()

        # Atualiza os valores de W utilizando o esquema escolhido
        if esquema == 1:
            for j in range(1,n):
                for i in range(1,m):
                    W[j][i] = (W[j][i+1] + W[j][i-1] + W[j+1][i] + W[j-1][i] + h*h*F[j][i])/4
        elif esquema == 2:
            for j in range(1,n):
                for i in range(1,m):
                    W[j][i] = (W[j+1][i+1] + W[j-1][i-1] + W[j+1][i-1] + W[j-1][i+1] + 2*h*h*F[j][i])/4
        elif esquema == 3:
            for j in range(1,n):
                for i in range(1,m):
                    W[j][i] = (W[j][i+1] + W[j][i-1] + W[j+1][i] + W[j-1][i])/5 +\
                              (W[j+1][i+1] + W[j-1][i-1] + W[j+1][i-1] + W[j-1][i+1])/20 +\
                               h*h*(F[j][i+1] + F[j][i-1] + F[j+1][i] + F[j-1][i] + 8*F[i][j])/40
        else:
            return None

        # Erro estimado (norma infinito)
        diff = np.max(np.abs(W-W_prev))

        # Plota "mapa de calor" da funcao aproximada
        if do_plot and p % plot_step == 0:
            plt.clf()
            plt.imshow(W)
            plt.title(f"Passo = {p+1}\nErro = {diff:.3}")
            plt.colorbar()
            plt.pause(0.00001)

        # Se erro estimado <= tolerancia, retorna
        if diff <= tol:
            break

    if do_plot:
        plt.clf()
        plt.imshow(W)
        plt.title(f"Passo = {p+1}\nErro = {diff:.3}")
        plt.colorbar()
        plt.show()

    if do_print:
        print(f"** FIM DA EXECUCAO **\nErro = {diff:.3}\nIteracao = {p+1}\n")
    return (x,y,W)

# Vizualizacao do metodo
esquema = 1
M = 40
x,y,W = scheme(m = M, n = M, esquema = esquema, max_iter = 3000, do_plot = 1, tol = 1e-5)

# Erro
X,Y = np.meshgrid(x,y)
print(f"Erro = {np.max(np.abs(np.flip(W,axis=1) - u(X,Y)))}")

plt.imshow(np.flip(u(X,Y),axis=1))
plt.colorbar()
plt.title("u(x,y)")
plt.show()

# Plot 3D da funcao aproximada

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X,Y,np.flip(W,axis=1),cmap=cm.coolwarm, antialiased=False)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("w(x,y)")
plt.show()

# Analise de erro + ordem de convergencia
esquema = 1
print()
print(f"          ** Esquema {esquema} **\n")
M = 5
E = [1]
for _ in range(4):

    x,y,W = scheme(m=M,n=M,esquema=esquema,max_iter=2500,tol=-1, do_plot = 0, do_print = 0)
    
    X,Y = np.meshgrid(x,y)

    # Erro em norma infinito
    E.append(np.max(np.abs(np.flip(W,axis=1) - u(X,Y))))
    div_err = E[-2]/E[-1]
    print(f"       m = n = {M}")
    print(f"       Erro = {E[-1]:.5}")
    print(f"       Erro^k/Erro^k+1 = {div_err:.5}")
    print(f"       Ordem \u2243 {np.log2(div_err):.5}")
    print()

    M *= 2

# ================================================================================================= #

# Analise do tempo computacional
def scheme_fast(m, n,
           esquema = 1,
           X = [0,1],
           Y = [0,1],
           max_iter = 1000):
    h = (X[1]-X[0])/m
    k = (Y[1]-Y[0])/n
    x = np.array([X[0] + i*h for i in range(m+1)])
    y = np.array([Y[0] + j*k for j in range(n+1)])
    W = np.zeros((n+1,m+1))
    F = [[f(x[i],y[j]) for i in range(m+1)] for j in range(n+1)]
    W[0, :] = u(Y[1],x)
    W[-1,:] = u(Y[0],x)
    W[:, 0] = u(y,X[1])
    W[:,-1] = u(y,X[0])
    for p in range(max_iter):
        if esquema == 1:
            for j in range(1,n):
                for i in range(1,m):
                    W[j][i] = (W[j][i+1] + W[j][i-1] + W[j+1][i] + W[j-1][i] + h*h*F[j][i])/4
        elif esquema == 2:
            for j in range(1,n):
                for i in range(1,m):
                    W[j][i] = (W[j+1][i+1] + W[j-1][i-1] + W[j+1][i-1] + W[j-1][i+1] + 2*h*h*F[j][i])/4
        elif esquema == 3:
            for j in range(1,n):
                for i in range(1,m):
                    W[j][i] = (W[j][i+1] + W[j][i-1] + W[j+1][i] + W[j-1][i])/5 +\
                              (W[j+1][i+1] + W[j-1][i-1] + W[j+1][i-1] + W[j-1][i+1])/20 +\
                               h*h*(F[j][i+1] + F[j][i-1] + F[j+1][i] + F[j-1][i] + 8*F[i][j])/40
        else:
            return None
    return (x,y,W)

R = 100
for esquema in (1,2,3):
    T = []
    H = []
    for M in (5,10,20,40):
        mean = 0
        for _ in range(R):
            timer = time.time()
            scheme_fast(M,M,esquema=esquema,max_iter=100)
            mean += time.time() - timer
        mean /= R
        T.append(mean)
        H.append(1/M)
    plt.plot(H, T, linestyle='dashed', label = f"Esquema {esquema}")
plt.title(f"Tempo x h")
plt.xlabel("h")
plt.ylabel("t(seg.)")
plt.legend()
plt.show()











