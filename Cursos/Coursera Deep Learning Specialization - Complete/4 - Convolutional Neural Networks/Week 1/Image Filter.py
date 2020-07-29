#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def vertical_edge(mode = 'bw'):

    ret_filter = np.array([
        [1,0,-1],
        [1,0,-1],
        [1,0,-1]
        ])
    if mode == 'bw':
        return ret_filter
    elif mode == 'rgb':
        return np.dstack((ret_filter,ret_filter,ret_filter))

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

def conv_forward(image, filtr, padding = 0, stride = 1):
    
    if len(image.shape) == 2:
        img = pad(image, padding, 'bw')
        m, n = img.shape
        f, g = filtr.shape
        size = (int(np.floor((m-f)/stride + 1)), int(np.floor((n-g)/stride + 1)))
        ret_matrix = np.zeros(size)
        for i in range(size[0]):
            for j in range(size[1]):
                ret_matrix[i,j] = np.sum(img[i*stride:i*stride+f, j*stride:j*stride+g] * filtr)
    elif len(image.shape) == 3:
        img = pad(image, padding, 'rgb')
        m, n, _ = img.shape
        f, g, _ = filtr.shape
        size = (int(np.floor((m-f)/stride + 1)), int(np.floor((n-g)/stride + 1)))
        ret_matrix = np.zeros(size)
        for i in range(size[0]):
            for j in range(size[1]):
                ret_matrix[i,j] = np.sum(img[i*stride:i*stride+f, j*stride:j*stride+g, :] * filtr)
        
    return ret_matrix

def pool_forward(image, f, stride, mode = 'max'):

    img = np.copy(image)

    if len(img.shape) == 2:
        m, n = img.shape
        size = (int(np.floor((m-f)/stride + 1)), int(np.floor((n-f)/stride + 1)))
        ret_matrix = np.zeros(size)
        for i in range(size[0]):
            for j in range(size[1]):
                if mode == 'max':
                    ret_matrix[i,j] = np.max(img[i*stride:i*stride+f, j*stride:j*stride+f])
                elif mode == 'average':
                    ret_matrix[i,j] = np.mean(img[i*stride:i*stride+f, j*stride:j*stride+f])
    elif len(img.shape) == 3:
        m, n, o = img.shape
        size = (int(np.floor((m-f)/stride + 1)), int(np.floor((n-f)/stride + 1)))
        ret_matrix = np.dstack((np.zeros(size), np.zeros(size), np.zeros(size)))
        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(o):
                    if mode == 'max':
                        ret_matrix[i,j,k] = np.max(img[i*stride:i*stride+f, j*stride:j*stride+f, k])
                    elif mode == 'mean':
                        ret_matrix[i,j,k] = np.mean(img[i*stride:i*stride+f, j*stride:j*stride+f, k])

    return ret_matrix

img = (np.random.random((100,100,3))*255).astype(np.uint8)
plt.imshow(img)
plt.show()

img1 = conv_forward(img, vertical_edge('rgb'), padding = 0, stride = 10)
plt.imshow(img1, cmap = "Greys")
plt.show()

img2 = pool_forward(img, f = 3, stride = 1, mode = 'mean')
plt.imshow((255*img2/np.max(img2)).astype(np.uint8))
plt.show()













