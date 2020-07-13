#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def vertical_edge():
    ret_filter = np.array([
        [1,0,-1],
        [1,0,-1],
        [1,0,-1]
        ])
    return  np.dstack((ret_filter,ret_filter,ret_filter))

def pad(image, p, mode):
    
    img = np.copy(image)
    if mode == 'bw':
        for _ in range(p):
            img = np.concatenate((img, np.zeros((img.shape[0], 1))), axis = 1)
            img = np.concatenate((np.zeros((img.shape[0], 1)), img), axis = 1)
            img = np.concatenate((img, np.zeros((1, img.shape[1]))), axis = 0)
            img = np.concatenate((np.zeros((1, img.shape[1])), img), axis = 0)
    elif mode == 'rgb':
        for _ in range(p):
            img = np.concatenate((img, np.zeros((img.shape[0], 1, 3))), axis = 1)
            img = np.concatenate((np.zeros((img.shape[0], 1, 3)), img), axis = 1)
            img = np.concatenate((img, np.zeros((1, img.shape[1], 3))), axis = 0)
            img = np.concatenate((np.zeros((1, img.shape[1], 3)), img), axis = 0)
    return img

def conv_forward(image, filtr, padding = 0, stride = 1, mode = 'bw'):
    
    img = pad(image, padding, mode)
    if mode == 'bw':
        m, n = img.shape
        f, g = filtr.shape
        size = (int(np.floor((m-f)/stride + 1)), int(np.floor((n-g)/stride + 1)))
        ret_matrix = np.zeros(size)
        for i in range(0, size[0]):
            for j in range(0, size[1]):
                ret_matrix[i,j] = np.sum(img[i:i+filtr.shape[0], j:j+filtr.shape[1]] * filtr)
    elif mode == 'rgb':
        m, n, _ = img.shape
        f, g, _ = filtr.shape
        size = (int(np.floor((m-f)/stride + 1)), int(np.floor((n-g)/stride + 1)))
        ret_matrix = np.zeros(size)
        for i in range(size[0]):
            for j in range(size[1]):
                ret_matrix[i,j] = np.sum(img[i:i+filtr.shape[0], j:j+filtr.shape[1], :] * filtr)
        
    return ret_matrix

img = (np.random.random((100,100,3))*255).astype(np.uint8)
plt.imshow(img)
plt.show()

img1 = conv_forward(img, vertical_edge(), padding = 10, stride = 1, mode = 'rgb')
plt.imshow(img1, cmap = "Greys")
plt.show()           



