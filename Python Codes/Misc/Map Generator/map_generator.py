

import time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc

directions_ = ("U","D","L","R")

class low_ground():
    def __init__(self):
        self.auto_prob = 0.9
    def look_around(self,look_for,map_,i,j,multiplier=1):
        points = 0
        try:
            if isinstance(map_[i+1][j], type(look_for)):
                points+=1
        except:
            pass
        if i-1 < 0:
            pass
        else:
            if isinstance(map_[i-1][j], type(look_for)):
                points+=1
        if j-1 < 0:
            pass
        else:
            if isinstance(map_[i][j-1], type(look_for)):
                points+=1
        try:
            if isinstance(map_[i][j+1], type(look_for)):
                points+=1
        except:
            pass
        self.auto_prob *= multiplier*points/3
    def spread(self, map_, i, j, decay = 0.7, tries = 4):
        for _ in range(tries):
            direct = np.random.choice(directions_)
            if direct == "U" and np.random.random() <= self.auto_prob:
                try:
                    if isinstance(map_[i+1][j], low_ground):
                        pass
                    else:
                        try:
                            map_[i+1][j] = low_ground()
                            map_[i+1][j].look_around(low_ground(),map_,i+1,j)
                            map_[i+1][j].auto_prob *= decay
                            map_ = map_[i+1][j].spread(map_,i+1,j,decay,tries)
                        except:
                            pass
                except:
                    pass
            if direct == "D" and np.random.random() <= self.auto_prob:
                if i-1 < 0 or isinstance(map_[i-1][j], low_ground):
                    pass
                else:
                    try:    
                        map_[i-1][j] = low_ground()
                        map_[i-1][j].look_around(low_ground(),map_,i-1,j)
                        map_[i-1][j].auto_prob *= decay
                        map_ = map_[i-1][j].spread(map_,i-1,j,decay,tries)
                    except:
                        pass
            if direct == "L" and np.random.random() <= self.auto_prob:
                if j-1 < 0 or isinstance(map_[i][j-1], low_ground):
                    pass
                else:
                    try:    
                        map_[i][j-1] = low_ground()
                        map_[i][j-1].look_around(low_ground(),map_,i,j-1)
                        map_[i][j-1].auto_prob *= decay
                        map_ = map_[i][j-1].spread(map_,i,j-1,decay,tries)
                    except:
                        pass
            if direct == "R" and np.random.random() <= self.auto_prob:
                try:
                    if isinstance(map_[i][j+1], low_ground):
                        pass
                    else:
                        try:
                            map_[i][j+1] = low_ground()
                            map_[i][j+1].look_around(low_ground(),map_,i,j+1)
                            map_[i][j+1].auto_prob *= decay
                            map_ = map_[i][j+1].spread(map_,i,j+1,decay,tries)
                        except:
                            pass
                except:
                    pass
        return map_ 
class mid_ground():
    def __init__(self):
        pass
class hig_ground():
    def __init__(self):
        pass
class mountain():
    def __init__(self):
        pass
class spring():
    def __init__(self):
        pass
class river():
    def __init__(self):
        pass
class lake():
    def __init__(self):
        pass
class ocean():
    def __init__(self):
        pass
class tree():
    def __init__(self):
        pass
class grass():
    def __init__(self):
        pass
class tall_grass():
    def __init__(self):
        pass

class terraformer():

    def __init__(self, size = (20,20)):
        self.map_ = np.zeros(size).tolist()

    def initialize(self, terrain = "ocean", island = "low_ground"):
        m,n = len(self.map_),len(self.map_[0])
        if not (terrain in ("ocean", "lake", "map")):
            island = 0
        if terrain == "ocean":
            for i in range(m):
                for j in range(n):
                    self.map_[i][j] = ocean()
        if terrain == "low_ground":
            for i in range(m):
                for j in range(n):
                    self.map_[i][j] = low_ground()
        if terrain == "map":
            pass
        if island:
            i = np.random.randint(low = int(m*0.35), high = int(m*0.65))
            j = np.random.randint(low = int(n*0.35), high = int(n*0.65))

            if island == "low_ground":
                self.map_[i][j] = low_ground()
                self.map_ = self.map_[i][j].spread(self.map_,i,j,decay = 0.48, tries = 14)
                
    def hole_fill(self, fill_in = "low_ground",iterations = 2):
        m,n = len(self.map_),len(self.map_[0])
        for _ in range(iterations):
            if fill_in == "low_ground":
                for i in range(m):
                    for j in range(n):
                        points = 0
                        try:
                            if isinstance(self.map_[i+1][j], low_ground):
                                points+=1
                        except:
                            pass
                        if i-1 < 0:
                            pass
                        else:
                            if isinstance(self.map_[i-1][j], low_ground):
                                points+=1
                        if j-1 < 0:
                            pass
                        else:
                            if isinstance(self.map_[i][j-1], low_ground):
                                points+=1
                        try:
                            if isinstance(self.map_[i][j+1], low_ground):
                                points+=1
                        except:
                            pass
                        if points/4 >= np.random.random():
                            self.map_[i][j] = low_ground()
                    
    def print_map(self):
        map__ = dc(self.map_)
        for i in range(len(map__)):
            for j in range(len(map__[0])):
                if isinstance(map__[i][j],ocean):
                    map__[i][j] = 0
                else:
                    map__[i][j] = 1
        plt.imshow(map__, interpolation = "spline16",cmap='coolwarm')
        plt.show()

import sys
sys.setrecursionlimit(4000)

for __ in range(10):
    god = terraformer(size = (300,300))
    god.initialize()

    for _ in range(3):
        god.initialize(terrain = "map")
    god.hole_fill(iterations = 3)
    god.print_map()



            

        




    
