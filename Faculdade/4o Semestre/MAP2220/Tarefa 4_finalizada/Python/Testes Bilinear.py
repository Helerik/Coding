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

# Function phi
def phi(X):
    return np.cos(X[0]/25)*np.sin(X[1]/25)

# Error test function
def f(x,y):
    return x + y

# Discretization of the function phi onto a matrix
def discrete_phi(n):
    x = -150
    y = -150
    H = 300/n
    with open("phi.txt", "w") as file:
        while y <= 150:
            while x <= 150:
                file.write("%f " %(phi([x, y])))
                x += H
            x = -150
            y += H
            file.write("\n")
    return

# Bilinear interpolation function
def bilinear(data, px, py, xmin, xmax, ymin, ymax):
    Q = data

    Hx = (xmax - xmin)/(len(Q[0]) - 1)
    Hy = (ymax - ymin)/(len(Q) - 1)

    i = abs(int((px - xmin)/Hx))
    j = abs(int((py - ymin)/Hx))

    x1 = int(xmin + i*Hx)
    y1 = int(ymin + j*Hy)
    x2 = x1 + Hx
    y2 = y1 + Hy

    
    fxy = (Q[j][i]*(x2 - px)*(y2 - py) + Q[j][i+1]*(px - x1)*(y2 - py) + Q[j+1][i]*(x2 - px)*(py - y1) + Q[j+1][i+1]*(px - x1)*(py - y1))/((x2 - x1)*(y2 - y1))

    return fxy

# Parameters:
##################
sample_size = 10
##################

discrete_phi(sample_size)
PHI = txt_to_list("phi.txt")


# Plot of the bilinear interpolated function phi
Z = []

x = np.arange(-150.0, 150.0, 0.5)
y = np.arange(-150.0, 150.0, 0.5)

for j in range(y.size):
    Z.append([])
    for i in range(x.size):
        Z[j].append(bilinear(PHI, x[i], y[-j-1], -150, 150, -150, 150))

im = imshow(Z,cmap=cm.jet)
colorbar(im)

locs, labels = plt.xticks()
labels = [(float(item)-300)/2 for item in locs]
plt.xticks(locs, labels)

locs, labels = plt.yticks()
labels = [(-float(item)+300)/2 for item in locs]
plt.yticks(locs, labels)

show()

#####################################
# Error test:
si = 1

data = []
for y in range(si+1):
    data.append([])
    for x in range(si+1):
        data[y].append(f(x/si, y/si))

Z = []

xp = np.arange(0, 1, 0.001)
yp = np.arange(0, 1, 0.001)

for j in range(yp.size):
    Z.append([])
    for i in range(xp.size):
        Z[j].append(bilinear(data, xp[i], yp[j], 0, si, 0, si))

im = imshow(Z,cmap=cm.jet)
colorbar(im)

locs, labels = plt.xticks()
labels = [(float(item))/1000 for item in locs]
plt.xticks(locs, labels)

locs, labels = plt.yticks()
labels = [(float(item))/1000 for item in locs]
plt.yticks(locs, labels)

show()

x = np.arange(0, 1, 0.001)
y = np.arange(0, 1, 0.001)
X,Y = meshgrid(x, y)
Z = f(X, Y)

im = imshow(Z,cmap=cm.jet)
colorbar(im)

locs, labels = plt.xticks()
labels = [(float(item))/1000 for item in locs]
plt.xticks(locs, labels)

locs, labels = plt.yticks()
labels = [(float(item))/1000 for item in locs]
plt.yticks(locs, labels)
show()

##fig, ax = plt.subplots()
##plt.xlim(-0.1,1.1)
##plt.ylim(-0.1,1.1)
##
##plt.plot([0,0],[-2,2], color = "black", lw = 1)
##plt.plot([1,1],[-2,2], color = "black", lw = 1)
##plt.plot([-2,2],[0,0], color = "black", lw = 1)
##plt.plot([-2,2],[1,1], color = "black", lw = 1)
##
##
##plt.plot(0,0, 'bo')
##plt.plot(0,1 ,'bo')
##plt.plot(1,0,'bo')
##plt.plot(1,1,'bo')
##
##plt.plot(0.5,0.5, 'rx')
##
##plt.text(0.05, 0.05, "(x{i}, y{j})")
##plt.text(0.75, 0.05, "(x{i+1} y{j})")
##plt.text(0.05, 0.95, "(x{i}, y{j+1})")
##plt.text(0.7, 0.95, "(x{i+1}, y{j+1})")
##plt.text(0.55,0.5, "(xp,yp)")
##ax.set_yticklabels([])
##ax.set_xticklabels([])
##plt.show()
##
##

