from math import cos, sin, e, pi, log, sqrt, factorial
import random as r
import time as t
import scipy as sp
from scipy.stats import gamma, norm
import matplotlib.pylab as plt
import numpy as np
import sys

# Passo de definicao

# VEC eh um vetor contendo todos os vetores xizes solicitados, como na tabela
VEC = []
VEC.append([1,20-3,2])
VEC.append([1,20-4,3])
VEC.append([1,20-5,4])
VEC.append([1,20-6,5])
VEC.append([1,20-7,6])
VEC.append([1,20-8,7])
VEC.append([1,20-9,8])
VEC.append([1,20-10,9])
VEC.append([1,20-11,10])
VEC.append([1,20-12,11])
VEC.append([1,20-13,12])
VEC.append([1,20-14,13])
VEC.append([1,20-15,14])
VEC.append([1,20-16,15])
VEC.append([1,20-17,16])
VEC.append([1,20-18,17])
VEC.append([1,20-19,18])
VEC.append([5,20-5,0])
VEC.append([5,20-6,1])
VEC.append([5,20-7,2])
VEC.append([5,20-8,3])
VEC.append([5,20-9,4])
VEC.append([5,20-10,5])
VEC.append([5,20-11,6])
VEC.append([5,20-12,7])
VEC.append([5,20-13,8])
VEC.append([5,20-14,9])
VEC.append([5,20-15,10])
VEC.append([9,20-9,0])
VEC.append([9,20-10,1])
VEC.append([9,20-11,2])
VEC.append([9,20-12,3])
VEC.append([9,20-13,4])
VEC.append([9,20-14,5])
VEC.append([9,20-15,6])
VEC.append([9,20-16,7])

# Nossa f para a hipotese, dado que a hipotese eh:
# - 1 = theta1 + theta2 + theta3
#   theta2 = 1 - theta1 - theta3
# - Theta = (theta1, theta2, theta3), Theta >= 0
# - theta3 = [1 - sqrt(theta1)]^2
#
# Podemos parametrizar a f somente para theta 1, da seguinte forma, onde x eh o vetor de xizes escolhidos de VEC:
def f(theta1, x):
    res = ((theta1)**(x[0])) * ((1 - theta1 - (1 - np.sqrt(theta1))**2 )**(x[1])) * (((1 - np.sqrt(theta1))**2)**(x[2]))
    return res

# 1 - Passo de otimizacao

# Grafico de todas as curvas, juntas
def print_graf1():
    t = np.arange(0., 1., 0.001)
    for i in range(len(VEC)):
        plt.plot(t,f(t,VEC[i]))
    plt.show()

#Graficos individuais das curvas
def print_graf2(i):
    t = np.arange(0., 1., 0.0005)
    arg = find_argmax(10000,VEC[i])[0]
    plt.plot(t,f(t,VEC[i]))
    plt.plot(arg,f(arg,VEC[i]),'o')
    plt.show()
        
# Enontra theta1 que maximiza a funcao, com n passos, para o dado vetor x:
def find_argmax(n,x):
    maxf = 0
    argmax = 0
    for i in range(n):
        if f(i/n,x) > maxf:
            maxf = f(i/n,x)
            argmax = i/n
    return [argmax, f(argmax,x)]

# Encontra theta*:
def find_theta(theta1):
    return [theta1, (1 - theta1 - (1 - np.sqrt(theta1))**2), ((1 - np.sqrt(theta1))**2)]

# 2 - Passo de integracao e calibragem

# Devemos definir nosso F, fora da hipotese, o qual sera utilizado para o MCMC. Para isso, temos as seguintes condicoes:
# - 1 = theta1 + theta2 + theta3
#   theta2 = 1 - theta1 - theta3
# - Theta = (theta1, theta2, theta3), Theta >= 0
#
# Onde x vai ser o conjunto de xizes utilizado, proveniente do VEC
def F(theta1,theta3,x):
    if theta1 > 1 or theta1 < 0 or theta3 > 1 or theta3 < 0:
        return 0
    if 1-theta1-theta3 <0:
        return 0
    res = ((theta1)**(x[0])) * ((1 - theta1 - theta3)**(x[1])) * ((theta3)**(x[2]))
    return res

# Nucleo Normal multivariado (duas variaveis)
def N2(mu,E):
    x = np.random.normal(mu[0],E[0])
    y = np.random.normal(mu[1],E[1])
    return [x,y]

# MCMC em duas dimensoes (duas variaveis)
def MCMC_2d(n, E, inicio, x):
    burn = int(0.2*n)
    n += burn
    dist = [0 for i in range(n)]
    dist[0] = inicio
    for i in range(n-1):
        proxX,proxY = N2(dist[i], E)
        if r.uniform(0,1) <= min(1, F(proxX,proxY,x)/F(dist[i][0],dist[i][1],x)):
            dist[i+1] = [proxX,proxY]
        else:
            dist[i+1] = dist[i]
    for i in range(burn):
        dist.pop(i)
    return dist

# Definimos nossa hipotese para 1 - integral, o e-valor da seguinte forma:
def ev(n, E, inicio, x):
    count = 0
    pontos = MCMC_2d(n,E,inicio, x)
    j = int(len(pontos)/100)
    print("-" + "-"*98 + "-")
    f_max = find_argmax(5000, x)[1]
    for i in range(len(pontos)):
        if i ==  j:
            print('\u2588', end='')
            j += int(len(pontos)/100)
        if F(pontos[i][0],pontos[i][1],x) > f_max:
            count += 1
    count = count/n
    print('\u2588')
    print()
    return 1 - count

# A autocorrelcao eh util para decidir se nosso passo esta bom, e se o erro sera baixo.
def autocorr(data):
    n = len(data)

    media = 0
    for i in range(n):
        media += data[i]
    media = media/n

    var = 0
    for i in range(n):
        var += (data[i]-media)**2
    var = var/n

    res = []
    soma = 0
    for k in range(n//2):
        for t in range(n-k):
            soma += (data[t]-media)*(data[t+k]-media)
        soma = soma/(var*(n-k))
        res.append(abs(soma))
    
    return res

def main():

    
    print()
    auto = str(input("Deseja fazer o calculo no modo automatico? [\u03C3s = 0.1, iteraceoes = 10000]: "))
    for i in range(len(VEC)):

        print()
        print("Resultados para x1 = %.0f, x2 = %.0f, x3 = %.0f: " %(VEC[i][0],VEC[i][1],VEC[i][2]))
        print()
        if auto == "s" or auto == "S":
            n = 10000
            E = [0.1,0.1]
        else:
            skip = str(input("Deseja pular esses xizes? [s/n]: "))
            print()
            if skip == "s" or skip == "S":
                pass
            else:
                while True:
                    
                    n = int(input("Digite o numero de iteracoes: "))
                    print()
                    print("Fase de definicao de \u03C3s:")
                    print()
                    E = []
                    E.append(float(input("Digite o \u03C31: ")))
                    E.append(float(input("Digite o \u03C32: ")))
                    print()
                    skip2 = str(input("Deseja pular a autocorrelacao? [s/n]: "))
                    if skip2 == "s" or skip2 == "S":
                        print("Continuar? [s/n]:",end=" ")
                        ok = input()
                        if ok == "s" or ok == "S":
                            break
                        else:
                            pass
                    else:
                        am = []
                        am[:] = MCMC_2d(n,E,[1/3,1/3],VEC[i])
                        U = []
                        for j in range(len(am)):
                            U.append(sqrt((am[j][0])**2 +(am[j][1])**2))
                        plt.plot(autocorr(U))
                        plt.show()
                        print("O resultado esta aceitavel? [s/n]:",end=" ")
                        ok = input()
                        if ok == "s" or ok == "S":
                            break
                        else:
                            pass
        print()
        print("Fase de calculo do e-valor:")

        tim = t.time()
        
        am = []
        am = MCMC_2d(n,E,[1/3,1/3],VEC[i])[:]
        x = []
        y = []
        for j in range(len(am)):
            x.append(am[j][0])
            y.append(am[j][1])
        print()
        print("e-valor =",ev(n,E,[1/3,1/3],VEC[i]))
        print("Tempo de computacao:", t.time() - tim)
        print()
        print("="*100)
##        print_graf2(i)
        plt.plot(x,y,'x')
        plt.show()
##    print_graf1()  

main()


##def density_plot(i):
##    am = []
##    am = MCMC_2d(500000,[0.1,0.1],[1/3,1/3],VEC[i])[:]
##    x = []
##    y = []
##    for j in range(len(am)):
##        x.append(am[j][0])
##        y.append(am[j][1])
##    plt.hist2d(x, y, bins=(150, 150), cmap=plt.cm.jet)
##    plt.show()
##
##                 
##
##for i in range(len(VEC)):
##    density_plot(i)























# MCMC exemplo, utilizando um paraboloide:
#
##def MCMC_2d_(n, E, inicio):
##    dist = [0 for i in range(n)]
##    dist[0] = inicio
##    for i in range(n-1):
##        proxX, proxY = N2(dist[i], E)
##        if r.uniform(0,1) <= min(1, func(proxX,proxY)/func(dist[i][0],dist[i][1])):
##            dist[i+1] = [proxX,proxY]
##        else:
##            dist[i+1] = dist[i]
##    return dist
##
##def func(x,y):
##    return 1 - (10*x**2) - (y**2)
##
##X = MCMC_2d_(10000,E,[0,0])
##x = []
##y = []
##for i in range(len(X)):
##    x.append(X[i][0])
##    y.append(X[i][1])
##plt.plot(x,y,'x')
##plt.show()
















