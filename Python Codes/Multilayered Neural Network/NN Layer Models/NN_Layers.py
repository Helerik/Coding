#!/usr/bin/env python3
# Author: Erik Davino Vincent

import sys
sys.path.insert(1, 'C:/Users/Cliente/Desktop/Coding/Python Codes/Multilayered Neural Network')

import numpy as np
from ActivationFunction import *

class Layer():
    '''
    Layer Class: Building block of a Neural Network

    ...

    Attributes
    ----------
    size : int
        the number of nodes this layer has
    activation : str
        name of activation function of the layer, e.g. "relu", "sigmoid"
    optimizer : object
        optimizer that will be used to update the weights and biases
    is_output : bool
        defines if this is an output layer (default False)
    Z : numpy.array
        holds the dot product of weights and previous activation/X
    A : numpy.array
        the activated value of Z
    dA_prev : numpy.array
        derivative/error of previous activation
    W : numpy.array
        the weights of this layer
    b : numpy.array
        the biases of this layer

    Methods
    -------
    init_weights(size_prev)
        initializes this layer's weights
    forward_pass(A_prev)
        performs forward propagation on this layer
    backward_pass(A_prev, optimizer, dA = None, Y = None)
        performs backward propagation on this layer
    '''
    
    def __init__(self,
                 size,
                 activation,
                 optimizer,
                 is_output = False):
        '''
        Parameters
        ----------
        size : int
            the number of nodes this layer has
        activation : str
            name of activation function of the layer, e.g. "relu", "sigmoid"
        optimizer : object
            optimizer that will be used to update the weights and biases
        is_output : bool
            defines if this is an output layer (default False)
        '''

        # Structural variables
        self.size = size
        self.optimizer = optimizer
        self.is_output = is_output

        if activation.lower() == 'sigmoid':
            self.activation = Sigmoid
        elif activation.lower() == 'tanh':
            self.activation = Tanh
        elif activation.lower() == 'ltanh':
            self.activation = LeakyTanh
        elif activation.lower() == 'relu':
            self.activation = ReLu
        elif activation.lower() == 'softmax':
            self.activation = Softmax

        # Cached variables
        self.Z = None
        self.A = None
        
        self.dA_prev = None

        self.W = None
        self.b = None

    def init_weights(self, size_prev):
        '''Initializes this layer's weights.
        
        Creates W and b using this layer's size and the previous layer's size.

        Parameters
        ----------
        size_prev : int
            the number of nodes on the previous layer
        '''

        self.W = np.random.randn(self.size, size_prev)*np.sqrt(2/size_prev)
        self.b = np.zeros((self.size, 1))    

    def forward_pass(self, A_prev):
        '''Performs forward propagation on this layer.

        Parameters
        ----------
        A_prev : numpy.array
            the activations of the previous layer
        '''

        W = self.W
        b = self.b

        Z = np.dot(W, A_prev) + b
        A = self.activation.function(Z)

        self.Z = Z
        self.A = A

    def backward_pass(self, A_prev, dA = None, Y = None):
        '''Performs backward propagation on this layer.

        If this is the output layer, dA is 'don't care' and Y must be passed.
        Otherwise, dA must be passed and Y is 'don't care'.
        The optimizer is an object that must contain an update method for W and b.

        Parameters
        ----------
        A_prev : numpy.array
            the activations of the previous layer
        dA : numpy.array
            the derivative for this layer's activations
        Y : numpy.array
            the ground truth labels
        '''
        
        Z = self.Z

        W = self.W
        b = self.b

        if self.is_output:
            dZ = self.A - Y
        else:
            dZ = dA * self.activation.derivative(Z)

        self.dA_prev = np.dot(W.T, dZ)

        W, b = self.optimizer.update(A_prev, dZ, W, b)

        self.W = W
        self.b = b
            
class GradientDescent():
    '''
    Gradient Descent: slow, but simple optimizer

    ...

    Attributes
    ----------
    learning_rate : float
        learning rate parameter - regulates gradient step size
    L2 : float
        L2-norm regularization; prevents overfitting
    

    Methods
    -------
    update(A_prev, dZ, W, b)
        updates W and b using a gradient descent step
    '''

    def __init__(self, learning_rate = 0.001, L2 = 0):

        self.learning_rate = learning_rate
        self.L2 = L2

    def update(self, A_prev, dZ, W, b):

        m = A_prev.shape[1]
        
        dW = (np.dot(A_prev, dZ.T) + self.L2*W.T)/m
        db = np.sum(dZ, axis = 1, keepdims = 1)/m

        W = W - self.learning_rate*dW.T
        b = b - self.learning_rate*db

        return (W, b)

class Adam():
    '''
    Adam: an efficient optimizer

    ...

    Attributes
    ----------
    learning_rate : float
        learning rate parameter - regulates gradient step size
    L2 : float
        L2-norm regularization; prevents overfitting (default = 0)
    beta1 : float
        momentum parameter; should be between 0 and 1 (default 0.9)
    beta2 : float
        second momentum (RMSprop) parameter; should be between 0 and 1 (default 0.999)
    epsilon : float
        a small value to prevent divisions by zero (default 1e-8)
    

    Methods
    -------
    update(A_prev, dZ, W, b)
        updates W and b using an Adam step - this optimizer uses momentum to speed
        up convergence to a minimum.
    '''

    def __init__(self, learning_rate = 0.001, L2 = 0, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):

        self.learning_rate = learning_rate
        self.L2 = L2
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.VdW = 0
        self.Vdb = 0
        self.SdW = 0
        self.Sdb = 0

    def update(self, A_prev, dZ, W, b):

        m = A_prev.shape[1]

        VdW = self.VdW
        Vdb = self.Vdb
        SdW = self.SdW
        Sdb = self.Sdb
        
        dW = (np.dot(A_prev, dZ.T) + self.L2*W.T)/m
        db = np.sum(dZ, axis = 1, keepdims = 1)/m

        VdW = self.beta1*VdW + (1 - self.beta1)*dW.T
        vdb = self.beta1*Vdb + (1 - self.beta1)*db
        SdW = self.beta2*SdW + (1 - self.beta2)*np.square(dW.T)
        Sdb = self.beta2*Sdb + (1 - self.beta2)*np.square(db)

        W = W - self.learning_rate*VdW/(np.sqrt(SdW) + self.epsilon)
        b = b - self.learning_rate*Vdb/(np.sqrt(Sdb) + self.epsilon)

        self.VdW = VdW
        self.Vdb = Vdb
        self.SdW = SdW
        self.Sdb = Sdb

        return (W, b)

        





    







        
