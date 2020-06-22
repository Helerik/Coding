# -*- coding: utf-8 -*-

# Tarefa 4

# Erik Davino Vincent; NUSP: 10736583

from math import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
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

# Calculates the norma2 of a vector V
def norma(V):
    sum = 0
    for i in range(len(V)):
        sum += (V[i])**2
    return np.sqrt(sum)

# Function phi
def phi(X):
    return np.cos(X[0]/25)*np.sin(X[1]/25)

# Function f, the speed function of the robot
def f(t, X, phi):
    ret = np.array( [ (10**(-3) - phi(X))*(X[0] - 100*cos(t))/(norma(X-100*np.array([cos(t), sin(t)]))), \
                    ( -(10**(-3)) - phi(X))*(X[1] - 100*sin(t))/(norma(X-100*np.array([cos(t), sin(t)]))) ] )
    return ret

# Sets a unifromily random distributed 2D vector as the startign position of the robot
def set_start():
    return np.array([np.random.uniform(-150,150), np.random.uniform(-150,150)])

# Runge kutta's method for solving ODEs
def runge(to, tf, Xo, n):
    
    h = (tf-to)/n

    # Saves the data in txt files
    with open("runge.txt", "w") as file:
    
        file.write("%f " %to)
        for i in range(len(Xo)):
            file.write("%f " %(Xo[i]))
        file.write("\n")

    # As a secondary output, an average of the distance travelled by the robot can be calculated
    with open("distance.txt", "w") as file:
        file.write("0 ")
        file.write("\n")
    
    dist = 0
    
    for _ in range(n):
        
        with open("runge.txt", "r") as file:
            data = [file.readlines()[-1]]
            info = data[0].split()
            file.close()
                
        t = float(info[0])
        X = np.array([float(info[1]), float(info[2])])
        
        k1 = f(t, X, phi)
        k2 = f(t + h/2, X + (h/2)*k1, phi)
        k3 = f(t + h/2, X + (h/2)*k2, phi)
        k4 = f(t + h, X + h*k3, phi)

        X_prox = X + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t_prox = t + h
        
        distance = np.sqrt( (X_prox[0] - X[0])**2 + (X_prox[1] - X[1])**2)
        dist += distance

        # The robot stops if it's travelled at least 850 meters
        if dist >= 850:
            return dist

        # The robot also stops if it hits the walls of omega (-150 m x and y or 150 m x and y)
        if X_prox[0] < -150 or X_prox[0] > 150 or X_prox[1] < -150 or X_prox[1] > 150:
            return dist

        with open("distance.txt", "a") as file:
            file.write("%f " %(dist))
            file.write("\n")

        with open("runge.txt", "a") as file:
            file.write("%f " %t_prox)
            for i in range(len(X_prox)):
                file.write("%f " %(X_prox[i]))
            file.write("\n")

    return dist

# Parameters:
##################
tempo = 500
n = 5000
L = 1 # The greater, the slower the animation speed
##################

# Execution of the ODE integration, using Runge-Kutta's method:
Xo = [10,70]
dist = runge(0, tempo, Xo, n)

x = []
y = []
t = []
spd = []
pl = txt_to_list("runge.txt")
for i in range(len(pl)):
    if i%5 == 0:
        x.append(pl[i][1])
        y.append(pl[i][2])
        t.append(pl[i][0])
for i in range(len(x)):
    spd.append(norma(f(t[i],[x[i],y[i]],phi))) # velocity information, which can be obtained from f, the speed vector of our robot

dis = txt_to_list("distance.txt")
dis = [dis[i][0] for i in range(len(dis))]

# Set boundaries for the plot:
if max(x) > 0:
    maxX = max(x)*1.1
else:
    maxX = max(x)*0.9
if min(x) > 0:
    minX = min(x)*0.9
else:
    minX = min(x)*1.1

if max(y) > 0:
    maxY = max(y)*1.1
else:
    maxY = max(y)*0.9
if min(y) > 0:
    minY = min(y)*0.9
else:
    minY = min(y)*1.1

# Robot's full trajectory graph:

fig, ax1 = plt.subplots()
plt.xlim(minX, maxX)
plt.ylim(minY, maxY)

plt.title("Distance Travelled: %f m \n Total time: %f s" %(dist, t[-1]))
plt.xlabel("x(t)")
plt.ylabel("y(t)")

plt.grid()
plt.plot(Xo[0], Xo[1], 'bo', label = "start")
plt.plot(x[-1], y[-1], 'ro', label = "end")
plt.plot(x, y, linestyle = (0, (1, 1)))
plt.legend()
plt.show()

# Robot's movement animation:
Writer = writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist=''), bitrate=3600)

fig, ax = plt.subplots()
plt.xlim(minX, maxX)
plt.ylim(minY, maxY)

plt.grid()
plt.title("Average speed: %f m/s" %(dist/t[-1]))
plt.xlabel("x(t)")
plt.ylabel("y(t)")

plt.plot(Xo[0], Xo[1], 'bo')
plt.plot(x[-1], y[-1], 'ro')

xdata, ydata = [], []
ln, = plt.plot([], [], 'r', animated=True, linestyle = (0, (1, 1)))
f = np.array([i for i in range(len(x))])

title = ax.text(0.5,0.80, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")

def init():
    ln.set_data(xdata,ydata)
    return ln,
    
def update(frame):
    xdata.append(x[frame])
    ydata.append(y[frame])
    try:
        title.set_text("time: %f s \n distance: %f m \n speed: %f m/s" %(t[frame], dis[frame], spd[frame]))
    except:
        title.set_text("time: %f s \n distance: %f m \n speed: %f m/s" %(t[frame], dis[frame], spd[frame]))
    
    ln.set_data(xdata, ydata)
    return ln, title

ani = FuncAnimation(fig, update, frames=f, init_func=init, blit=True, interval = tempo/(L*n) ,repeat=False)
ani.save("MyAnimation.mp4", writer = writer, dpi = 300)
plt.show()	






##x = np.arange(-150.0, 150.0, 0.1)
##y = np.arange(-150.0, 150.0, 0.1)
##X,Y = meshgrid(x, y)
##Z = phi([X, -Y])
##
##im = imshow(Z,cmap=cm.jet)
##colorbar(im)
##
##locs, labels = plt.xticks()
##labels = [(float(item)-1500)/10 for item in locs]
##plt.xticks(locs, labels)
##locs, labels = plt.yticks()
##labels = [(-float(item)+1500)/10 for item in locs]
##plt.yticks(locs, labels)
##show()
##


