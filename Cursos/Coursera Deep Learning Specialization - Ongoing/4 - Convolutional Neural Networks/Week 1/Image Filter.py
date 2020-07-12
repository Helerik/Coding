#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def vertical_edge():
    ret_filter = np.array([
        [1,0,-1],
        [1,0,-1],
        [1,0,-1]
        ])
    return  ret_filter

def conv_forward(img, filtr):
    ret_matrix = np.zeros((img.shape[0]-filtr.shape[0]+1, img.shape[1]-filtr.shape[1]+1))
    for i in range(img.shape[0]-filtr.shape[0]+1):
        for j in range(img.shape[1]-filtr.shape[1]+1):
            ret_matrix[i,j] = np.sum(img[i:i+filtr.shape[0], j:j+filtr.shape[1]] * filtr)
    return ret_matrix        

img = np.array([
    [3,3,3,3,3,0,0,0,0],
    [3,3,3,3,3,0,0,0,0],
    [3,3,3,3,3,0,0,0,0],
    [3,3,3,3,3,0,0,0,0],
    [3,3,3,3,3,0,0,0,0],
    [3,3,3,3,3,0,0,0,0],
    [3,3,3,3,3,0,0,0,0],
    [3,3,3,3,3,0,0,0,0],
    [3,3,3,3,3,0,0,0,0]
    ])

img1 = conv_forward(img, vertical_edge())

plt.imshow(img, cmap = "Greys")
plt.show()

plt.imshow(img1, cmap = "Greys")
plt.show()           
