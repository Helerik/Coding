# -*- coding: utf-8 -*-

# Tarefa 4

# Erik Davino Vincent; NUSP: 10736583

from math import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

# Converts a txt archive to a python list
def txt_to_list(txt):
    f = open(txt, "r")
    ret = []
    data = f.readlines()
    for line in data:
        words = line.split()
        ret.append([float(word) for word in words])
    return ret

# Function f = F'
def f(t, X):
    ret = np.array( [ X[0]*t, \
                      X[1]*t ] )
    return ret

def F(t, X):
    return [np.exp((t**2)/2), np.exp((t**2)/2)]

# Runge kutta's method for solving ODEs
def runge(to, tf, Xo, n):
    
    h = (tf-to)/n

    # Saves the data in txt files
    with open("runge.txt", "w") as file:
    
        file.write("%f " %to)
        for i in range(len(Xo)):
            file.write("%f " %(Xo[i]))
        file.write("\n")
    
    for _ in range(n):
        
        with open("runge.txt", "r") as file:
            data = [file.readlines()[-1]]
            info = data[0].split()
            file.close()
                
        t = float(info[0])
        X = np.array([float(info[1]), float(info[2])])
        
        k1 = f(t, X)
        k2 = f(t + h/2, X + (h/2)*k1)
        k3 = f(t + h/2, X + (h/2)*k2)
        k4 = f(t + h, X + h*k3)

        X_prox = X + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t_prox = t + h

        with open("runge.txt", "a") as file:
            file.write("%f " %t_prox)
            for i in range(len(X_prox)):
                file.write("%f " %(X_prox[i]))
            file.write("\n")

# Parameters:
##################
tempo = 2
n = 2
##################

# Execution of the ODE integration, using Runge-Kutta's method and plot:
Xo = [1,1]

plt.figure()
plt.grid()

plt.xlabel("t")
plt.ylabel("Fx(t) = Fy(t)")

u = np.arange(0,2.1,0.1)
plt.plot(u, F(u, 1)[0], label = "F(t)")

err = ["n/d"]

for _ in range(9):
    
    runge(0, tempo, Xo, n)

    x = []
    y = []
    t = []
    pl = txt_to_list("runge.txt")
    for i in range(len(pl)):
        x.append(pl[i][1])
        y.append(pl[i][2])
        t.append(pl[i][0])
    err.append(abs(F(t[-1], x[-1])[0] - x[-1]))

    if _ == 0:
        print((2/n), "&", err[_], "&", "n/d","&", "n/d")
    elif _ == 1:
        print((2/n), "&", err[_], "&", "n/d","&", "n/d")
    else:
        print((2/n), "&", err[_],"&", err[ _ -1]/err[_] ,"&", "p = %f" %(np.log(err[_ - 1]/err[_])/np.log(2)))
    
    plt.plot(t, x, linestyle = (0,(1,1)), label = "Dx = %f" %(2/n))
    n*=2
plt.legend()
plt.show()

# Error plot

plt.figure()
plt.grid()
plt.plot(err[1:])
plt.title("Erro para cada iteracao")
plt.xlabel("Iteracao")
plt.ylabel("Erro")
plt.show()

pl = None
x = None
y = None
t = None
