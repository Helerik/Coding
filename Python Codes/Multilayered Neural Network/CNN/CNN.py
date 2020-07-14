#!/usr/bin/env python3
# Author: Erik Davino Vincent

# Imports sys for creating path to imported files on runtime
import sys
sys.path.insert(1, 'C:/Users/Cliente/Desktop/Coding/Python Codes/Multilayered Neural Network/')

import numpy as np
import matplotlib.pyplot as plt

# Pynput library for early stopping on command
from pynput import keyboard

from ActivationFunction import *

class CNN():

    # Initializes Neural Network structure
    def __init__(self,
                 layer_sizes = [
                     {'type':'conv', 'f_H':3, 'f_W':3, 'n_C':6, 'stride':1, 'pad':1},
                     {'type':'pool', 'f_H':14, 'f_W':14, 'n_C':8, 'stride':14, 'mode':'max'},
                     {'type':'conv', 'f_H':3, 'f_W':3, 'n_C':10, 'stride':1, 'pad':1},
                     {'type':'pool', 'f_H':3, 'f_W':3, 'n_C':12, 'stride':3, 'mode':'max'},
                     {'type':'fc', 'size':10},
                     {'type':'fc', 'size':10}
                     ],
                 
                 learning_rate = 0.01,
                 max_iter = 200,
                 L2 = 0,
                 beta1 = 0.9,
                 beta2 = 0.999,
                 epsilon = 1e-8,
                 
                 activation = "relu",
                 minibatch_size = None,
                 classification = "multiclass",
                 
                 plot_N = None,
                 end_on_close = False,
                 end_on_backspace = False):

        # Structural variables
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) + 1
        
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.L2 = L2
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.minibatch_size = minibatch_size
        self.classification = classification.lower()
        
        self.plot_N = plot_N
        self.end_on_close = end_on_close
        self.end_on_backspace = end_on_backspace

        # Activation function can be string or list
        if isinstance(activation, str):
            if activation.lower() == 'sigmoid':
                self.activation = [Sigmoid for _ in range(self.num_layers-1)]
                self._activation_str = ['sigmoid' for _ in range(self.num_layers-1)]
            elif activation.lower() == 'tanh':
                self.activation = [Tanh for _ in range(self.num_layers-1)]
                self._activation_str = ['tanh' for _ in range(self.num_layers-1)]
            elif activation.lower() == 'ltanh':
                self.activation = [LeakyTanh for _ in range(self.num_layers-1)]
                self._activation_str = ['ltanh' for _ in range(self.num_layers-1)]
            elif activation.lower() == 'relu':
                self.activation = [ReLu for _ in range(self.num_layers-1)]
                self._activation_str = ['relu' for _ in range(self.num_layers-1)]
        else:
            self._activation_str = np.copy(activation)
            for i in range(len(activation)):
                if activation[i].lower() == 'sigmoid':
                    activation[i] = Sigmoid
                elif activation[i].lower() == 'tanh':
                    activation[i] = Tanh
                elif activation[i].lower() == 'ltanh':
                    activation[i] = LeakyTanh
                elif activation[i].lower() == 'relu':
                    activation[i] = ReLu
            self.activation = activation

        # Variable variables
        self.X = None
        self.Y = None
        self.m = None

        self.minibatches = None
        self.minibatch_X = None
        self.minibatch_Y = None
        self.minibatch_m = None
        self.best_minibatch_cost = np.inf

        self.fc_weights = False

        self.weights = {}
        self.best_weights = {}
        self.A_vals = {}
        self.Z_vals = {}
        self.V_vals = {}
        self.S_vals = {}

        self.training_status = "Untrained"

        self.code_breaker = 0

    def __str__(self):
        return f"""         {self.classification.capitalize()} Neural Network ({self.training_status}):

    |    Layer Size Structure:          {np.array(self.layer_sizes)}
    |    Activation Function Structure: {self._activation_str}
    |
    |    Learning Rate:                 {self.learning_rate}
    |    L2 Regularization:             {self.L2}
    |
    |    Beta1 (Momentum):              {self.beta1}
    |    Beta2 (RMSprop):               {self.beta2}
    |    Epsilon:                       {self.epsilon:.1e}
    |
    |    Mini-Batch Size:               {self.minibatch_size}
    |    Max Iterations:                {self.max_iter}"""

    # Initializes weights for each layer
    def __initialize_weights(self, n_Cx):

        # Convolutional and pooling weights
        n_C_prev = n_Cx
        for i in range(self.num_layers - 1):

            if self.layer_sizes[i]['type'] == 'conv':

                n_C = self.layer_sizes[i]['n_C']
                f_H = self.layer_sizes[i]['f_H']
                f_W = self.layer_sizes[i]['f_W']
                
                self.weights['W'+str(i+1)] = np.random.randn(n_C, n_C_prev, f_H, f_W)*np.sqrt(2/n_C)
                self.weights['b'+str(i+1)] = np.zeros((n_C,1,1,1))

                n_C_prev = n_C

            elif self.layer_sizes[i]['type'] == 'pool':
                self.weights['W'+str(i+1)] = None
                self.weights['b'+str(i+1)] = None

            elif self.layer_sizes[i]['type'] == 'fc':
                break
            
    # Initialize fully connected weights
    def __initialize_fc_weights(self, n_a, n_y):

        # Fully connected weights
        n_h_prev = n_a
        for i in range(self.num_layers - 1):

            if self.layer_sizes[i]['type'] == 'fc':
        
                n_h = self.layer_sizes[i]['size']
                
                self.weights['W'+str(i+1)] = np.random.randn(n_h, n_h_prev)*np.sqrt(2/n_h_prev)
                self.weights['b'+str(i+1)] = np.zeros((n_h,1))
                
                n_h_prev = n_h

        # Output weights
        self.weights['W'+str(self.num_layers)] = np.random.randn(n_y, n_h_prev)*np.sqrt(2/n_h_prev)
        self.weights['b'+str(self.num_layers)] = np.zeros((n_y,1))

    # Initializes momentum and RMSprop for each layer
##    def __initialize_momentum(self, n_Cx, n_y):
##
##        # Convolutional and pooling momentums
##        n_C_prev = n_Cx
##        for i in range(self.num_layers - 1):
##
##            if self.layer_sizes[i]['type'] == 'conv' or self.layer_sizes[i]['type'] == 'pool':
##
##                n_C = self.layer_sizes[i]['n_C']
##                f_H = self.layer_sizes[i]['f_H']
##                f_W = self.layer_sizes[i]['f_W']
##
##                self.V_vals["VdW"+str(i+1)] = np.zeros((n_C, n_C_prev, f_H, f_W))
##                self.V_vals["Vdb"+str(i+1)] = np.zeros((n_C,1,1,1))
##            
##                self.S_vals["SdW"+str(i+1)] = np.zeros((n_C, n_C_prev, f_H, f_W))
##                self.S_vals["Sdb"+str(i+1)] = np.zeros((n_C,1,1,1))
##
##                n_C_prev = n_C
##
##            else:
##                k = i
##                break
##
##        # Fully connected momentums
##        n_h_prev = len(self.V_vals["VdW"+str(k-1)].flatten())
##        for i in range(k, self.num_layers - 1):
##        
##            n_h = self.layer_sizes[i]['size']
##            
##            self.V_vals["VdW"+str(i+1)] = np.zeros((n_h, n_h_prev))
##            self.V_vals["Vdb"+str(i+1)] = np.zeros((n_h,1))
##            
##            self.S_vals["SdW"+str(i+1)] = np.zeros((n_h, n_h_prev))
##            self.S_vals["Sdb"+str(i+1)] = np.zeros((n_h,1))
##            
##            n_h_prev = n_h
##
##        # Output momentums
##        self.weights['W'+str(self.num_layers)] = np.random.randn(n_y, n_h_prev)*np.sqrt(2/n_h_prev)
##        self.weights['b'+str(self.num_layers)] = np.zeros((n_y,1))

    # Performs foward propagation
    def __forward_propagation(self):

        Ai_prev = self.minibatch_X.copy()
        for i in range(self.num_layers - 1):

            if self.layer_sizes[i]['type'] == 'conv':

                stridei = self.layer_sizes[i]['stride']
                padi = self.layer_sizes[i]['pad']
            
                Wi = self.weights['W'+str(i+1)].copy()
                bi = self.weights['b'+str(i+1)].copy()

                Zi = ConvPool.conv_forward(Ai_prev, Wi, bi, stridei, padi)
                Ai = self.activation[i].function(Zi)

                self.A_vals['A'+str(i+1)] = Ai.copy()
                self.Z_vals['Z'+str(i+1)] = Zi.copy()

                Ai_prev = Ai.copy()

            elif self.layer_sizes[i]['type'] == 'pool':

                f_Hi = self.layer_sizes[i]['f_H']
                f_Wi = self.layer_sizes[i]['f_W']
                stridei = self.layer_sizes[i]['stride']
                modei = self.layer_sizes[i]['mode']

                Ai = ConvPool.pool_forward(Ai_prev, f_Hi, f_Wi, stridei, modei)

                self.A_vals['A'+str(i+1)] = Ai.copy()
                self.Z_vals['Z'+str(i+1)] = None

                Ai_prev = Ai.copy()

            elif self.layer_sizes[i]['type'] == 'fc':
                k = i
                break

        Ai_prev = np.array([Ai.flatten()]).T
        
        if not self.fc_weights:
            n_a, m_a = Ai_prev.shape
            n_y, m_y = self.Y.shape
            self.__initialize_fc_weights(n_a, n_y)
            
        for i in range(k, self.num_layers - 1):

            Wi = self.weights['W'+str(i+1)].copy()
            bi = self.weights['b'+str(i+1)].copy()
            
            Zi = np.dot(Wi, Ai_prev) + bi
            Ai = self.activation[i].function(Zi)       

            self.A_vals['A'+str(i+1)] = Ai.copy()
            self.Z_vals['Z'+str(i+1)] = Zi.copy()
            
            Ai_prev = Ai.copy()

        Wi = self.weights['W'+str(self.num_layers)].copy()
        bi = self.weights['b'+str(self.num_layers)].copy()

        # Last layer always receives sigmoid or softmax
        Zi = np.dot(Wi, Ai_prev) + bi
        if self.classification == 'binary':
            Ai = Sigmoid.function(Zi)
        elif self.classification == 'multiclass':
            Ai = Softmax.function(Zi)

        self.A_vals['A'+str(self.num_layers)] = Ai.copy()
        self.Z_vals['Z'+str(self.num_layers)] = Zi.copy()

    # Performs backward propagation loop
    def __backward_propagation(self):

        # Initialize m
        m = self.minibatch_m
        
        for i in range(self.num_layers, 0, -1):

            # Gets Ai_prev. If i = 1, Ai_prev = X
            if i == 1:
                Ai_prev = self.minibatch_X.copy()
            else:
                Ai_prev = self.A_vals['A'+str(i-1)].copy()

            # Gets Zi value
            Zi = self.Z_vals['Z'+str(i)].copy()

            if self.layer_sizes[i]['type'] == 'fc':
            
                if i == self.num_layers:
                    AL = self.A_vals['A'+str(i)].copy()
                    dZi = AL - self.minibatch_Y
                else:
                    dZi = dAi * self.activation[i-1].derivative(Zi)

                # Gets current layer weights
                Wi = self.weights['W'+str(i)].copy()
                bi = self.weights['b'+str(i)].copy()

                # Gets momentum
##                VdWi = self.V_vals["VdW"+str(i)].copy()
##                Vdbi = self.V_vals["Vdb"+str(i)].copy()

                # Gets RMSprop
##                SdWi = self.S_vals["SdW"+str(i)].copy()
##                Sdbi = self.S_vals["Sdb"+str(i)].copy()
                
                # Cache dA; on last layer, dA = Wi.T . dZi = Wi.T . (Ai - Y)
                dAi = np.dot(Wi.T, dZi)

                # Calculates dWi and dbi
                dWi = np.dot(Ai_prev, dZi.T)/m + (self.L2/m)*Wi.T
                dbi = np.sum(dZi, axis = 1, keepdims = 1)/m

            elif self.layer_sizes[i]['type'] == 'pool':

                f_Hi = self.layer_sizes[i]['f_H']
                f_Wi = self.layer_sizes[i]['f_W']
                stridei = self.layer_sizes[i]['stride']
                modei = self.layer_sizes[i]['mode']

                dAi = ConvPool.pool_backward(dAi, Ai_prev, f_Hi, f_Wi, stridei, modei)

            elif self.layer_sizes[i]['type'] == 'conv':

                stridei = self.layer_sizes[i]['stride']
                padi = self.layer_sizes[i]['pad']
            
                Wi = self.weights['W'+str(i)].copy()
                bi = self.weights['b'+str(i)].copy()

                dAi = ConvPool.conv_backward(dZi, Ai_prev, Wi, bi, stridei, padi)
                dZi = dAi * self.activation[i-1].derivative(Zi)

            if self.layer_sizes[i]['type'] != 'pool':
##                # Updates momentum
##                VdWi = self.beta1*VdWi + (1-self.beta1)*dWi.T
##                Vdbi = self.beta1*Vdbi + (1-self.beta1)*dbi
##                self.V_vals["VdW"+str(i)] = VdWi.copy()
##                self.V_vals["Vdb"+str(i)] = Vdbi.copy()
##
##                # Updates RMSprop
##                SdWi = self.beta2*SdWi + (1-self.beta2)*np.square(dWi.T)
##                Sdbi = self.beta2*Sdbi + (1-self.beta2)*np.square(dbi)
##                self.S_vals["SdW"+str(i)] = SdWi.copy()
##                self.S_vals["Sdb"+str(i)] = Sdbi.copy()

                # Updates weights and biases
                Wi = Wi - self.learning_rate*dWi    #*VdWi/(np.sqrt(SdWi) + self.epsilon)
                bi = bi - self.learning_rate*dbi    #*Vdbi/(np.sqrt(Sdbi) + self.epsilon)
                self.weights['W'+str(i)] = Wi.copy()
                self.weights['b'+str(i)] = bi.copy()
                

    # Evaluates cost function
    def __evaluate_cost(self):
        
        AL = self.A_vals['A'+str(self.num_layers)].copy()

        if self.classification == 'binary':
            loss_func = -(self.minibatch_Y*np.log(AL) + (1-self.minibatch_Y)*np.log(1-AL))
        elif self.classification == 'multiclass':
            loss_func = -np.sum(self.minibatch_Y*np.log(AL), axis = 0, keepdims = 1)
        cost_func = np.mean(loss_func)

        # Evaluates regularization cost
        if self.L2 > 0:
            L2_reg = 0
            for i in range(1, self.num_layers):
                L2_reg += np.sum(np.square(self.weights['W'+str(i)]))
            L2_reg *= self.L2/(2*self.minibatch_m)
            cost_func += L2_reg

        if cost_func < self.best_minibatch_cost:
            self.best_minibatch_cost = cost_func

    # Creates minibatches
    def __make_minibatches(self):

        # Initializes list of minibatches
        self.minibatches = []

        # Shuffles data before creating minibatches
        permutation = list(np.random.permutation(self.m))
        shuffled_X = self.X[permutation]
        shuffled_Y = self.Y[:, permutation]

        num_full_minibatches = int(np.floor(self.m/self.minibatch_size))
        for k in range(num_full_minibatches):
            minibatch_X = shuffled_X[self.minibatch_size*k : self.minibatch_size*(k+1)]
            minibatch_Y = shuffled_Y[:, self.minibatch_size*k : self.minibatch_size*(k+1)]
            minibatch = (minibatch_X, minibatch_Y)
            self.minibatches.append(minibatch)

        # Depending on size of X and size of minibatches, the last minibatch will be smaller
        if self.m % self.minibatch_size != 0:
            minibatch_X = shuffled_X[self.minibatch_size*num_full_minibatches :]
            minibatch_Y = shuffled_Y[:, self.minibatch_size*num_full_minibatches :]
            minibatch = (minibatch_X, minibatch_Y)
            self.minibatches.append(minibatch)  

    # Newton method for training a neural network
    def __mini_batch_gradient_descent(self):
        
        # Cache for plotting cost
        if self.plot_N != None and self.plot_N != 0:
            fig = plt.figure()

            # Ads closing graph handler if end_on_close
            if self.end_on_close:
                def handle_close(evt):
                    self.code_breaker = 1
                fig.canvas.mpl_connect('close_event', handle_close)
            cost = []
            iteration = []

        # Ads close on command if end_on_backspace
        if self.end_on_backspace:
            def on_press(key):
                if key == keyboard.Key.backspace:
                    self.code_breaker = 1
            def on_release(key):
                pass
            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            listener.start()
            
        # Cache for best cost and best weights
        self.best_weights = self.weights.copy()
        best_cost = np.inf

        for it in range(self.max_iter):

            # Creates iteration minibatches and best minibatch cost
            self.__make_minibatches()
            self.best_minibatch_cost = np.inf

            if self.code_breaker:
                break
                
            # Performs operations on every minibatch
            for minibatch in self.minibatches:

                # Defines current minibatch
                self.minibatch_X = minibatch[0]
                self.minibatch_Y = minibatch[1]
                self.minibatch_m = self.minibatch_X.shape[1]
                m = self.minibatch_m

                # Forward propagation
                self.__forward_propagation()

                # Evaluate cost
                self.__evaluate_cost()
                
                # Updates best weights and best_cost
                if self.best_minibatch_cost < best_cost:
                    self.best_weights = self.weights.copy()
                    best_cost = self.best_minibatch_cost

                # Backward propagation
                self.__backward_propagation()

                if self.code_breaker:
                    break
                
            # Caches cost function every plot_N iterations and plots cost function over time
            if self.plot_N != None and self.plot_N != 0:
                if (it + 1) % self.plot_N == 0:
                    
                    cost.append(self.best_minibatch_cost)
                    iteration.append(it+1)
                    
                    plt.clf()
                    plt.plot(iteration, cost, color = 'b')
                    plt.xlabel("Iteration")
                    plt.ylabel("Cost Function")
                    plt.title(f"Cost Function over {iteration[-1]} iterations:")
                    plt.pause(0.001)

            if self.code_breaker:
                break

        # Final plot
        if self.plot_N != None and self.plot_N != 0:
            
            # Final cost evaluation:
            self.__evaluate_cost()
            cost.append(self.best_minibatch_cost)
            iteration.append(it+1)
            
            # Plots cost function over time
            plt.clf()
            plt.plot(iteration, cost, color = 'b')
            plt.xlabel("Iteration")
            plt.ylabel("Cost Function")
            plt.title(f"Cost Function over {iteration[-1]} iterations:")
            plt.show(block = 0)

        # Reset variables
        self.X = None
        self.Y = None
        self.m = None

        self.minibatches = None
        self.minibatch_X = None
        self.minibatch_Y = None
        self.minibatch_m = None
        self.best_minibatch_cost = np.inf

        self.weights = {}
        self.A_vals = {}
        self.Z_vals = {}
        self.V_vals = {}
        self.S_vals = {}

    # Sets the network up and fits for X and Y
    def fit(self, X, Y, warm_start = False):

        self.X = X
        if self.classification == 'binary':
            self.Y = Y
        elif self.classification == 'multiclass':
            self.Y = []
            class_num = np.max(Y)
            for i in range(Y.shape[1]):
                self.Y.append([0 for _ in range(class_num+1)])
                self.Y[i][Y[0][i]] = 1
            self.Y = np.array(self.Y).T
        
        n_x, m_x = self.X.shape
        n_y, m_y = self.Y.shape
        if m_x != m_y:
            raise ValueError(f"Invalid vector sizes for X and Y -> X size = {X.shape} while Y size = {Y.shape}.")

        if warm_start and self.training_status == "Trained":
            self.weights = self.best_weights
        else:
            self.__initialize_weights(n_x, n_y)
        self.__initialize_momentum(n_x, n_y)
            
        self.m = m_x

        if self.minibatch_size == None:
            self.minibatch_size = m_x

        # Trains the network
        self.__mini_batch_gradient_descent()
        self.training_status = "Trained"

    # Predicts X vector tags
    def predict(self, X):

        Ai_prev = X.copy()
        for i in range(self.num_layers - 1):
            
            Wi = self.best_weights['W'+str(i+1)].copy()
            bi = self.best_weights['b'+str(i+1)].copy()
            
            Zi = np.dot(Wi, Ai_prev) + bi
            Ai = self.activation[i].function(Zi)

            Ai_prev = Ai.copy()

        Wi = self.best_weights['W'+str(self.num_layers)].copy()
        bi = self.best_weights['b'+str(self.num_layers)].copy()

        # Last layer always receives sigmoid or softmax
        Zi = np.dot(Wi, Ai_prev) + bi
        if self.classification == 'binary':
            Ai = Sigmoid.function(Zi)
            return Ai > 0.5

        elif self.classification == 'multiclass':
            Ai = Softmax.function(Zi)
            prediction = []
            for i in range(Ai.shape[1]):
                prediction.append(np.argmax(Ai[:,i]))
            return np.array([prediction])

# Class for convolution and pooling operations + padding
class ConvPool():

    @classmethod
    def zero_pad(cls, A_prev, p):
        return np.pad(A_prev, ((0, 0), (p, p), (p, p), (0, 0)), 'constant', constant_values = (0, 0))

    @classmethod
    def __conv_step(cls, A_prev_slice, filt, add):
        return float(np.sum(A_prev_slice * filt) + add)

    @classmethod
    def conv_forward(cls, A_prev, W, b, stride, pad):

        (m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape
        (n_C, n_C_prev, f_H, f_W) = W.shape

        n_H = int(np.floor((n_H_prev-f_H+2*pad)/stride + 1))
        n_W = int(np.floor((n_W_prev-f_W+2*pad)/stride + 1))

        Z = np.zeros((m, n_C, n_H, n_W))

        A_prev_pad = cls.zero_pad(A_prev, pad)

        for i in range(m):
            A_previ_pad = A_prev_pad[i]
            for h in range(n_H):
                start_H = h*stride
                end_H = h*stride + f_H
                for w in range(n_W):
                    start_W = w*stride
                    end_W = w*stride + f_W
                    for c in range(n_C):

                        A_previ_slice = A_previ_pad[:, start_H:end_H, start_W:end_W]
                        W_filt = W[c,:,:,:]
                        b_filt = b[c,:,:,:]
                        Z[i, c, h, w] = cls.__conv_step(A_previ_slice, W_filt, b_filt)

        return Z

    @classmethod
    def pool_forward(cls, A_prev, f_H, f_W, stride, mode):

        (m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape

        n_H = int(np.floor((n_H_prev-f_H+2*pad)/stride + 1))
        n_W = int(np.floor((n_W_prev-f_W+2*pad)/stride + 1))
        n_C = n_C_prev

        A = np.zeros((m, n_C, n_H, n_W))

        for i in range(m):
            for h in range(n_H):
                start_H = h*stride
                end_H = h*stride + f_H
                for w in range(n_W):
                    start_W = w*stride
                    end_W = w*stride + f_W
                    for c in range(n_C):

                        A_slice = A[i, c, start_H:end_H, start_W:end_W]

                        if mode == "max":
                            A[i, c, h, w] = np.max(A_slice)
                        elif mode == "average":
                            A[i, c, h, w] = np.mean(A_slice)

        return A

    @classmethod
    def conv_backward(cls, dZ, A_prev, W, b, stride, pad):
        
        (m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape
        (n_C, n_C_prev, f_H, f_W) = W.shape
        (m, n_C, n_H, n_W) = dZ.shape

        dA_prev = np.zeros((m, n_C_prev, n_H_prev, n_W_prev))
        dW = np.zeros((n_C, n_C_prev, f_H, f_W))
        db = np.zeros((n_C, 1, 1, 1))

        Z = np.zeros((m, n_C, n_H, n_W))

        A_prev_pad = cls.zero_pad(A_prev, pad)
        dA_prev_pad = cls.zero_pad(dA_prev, pad)

        for i in range(m):
            A_previ_pad = A_prev_pad[i]
            dA_previ_pad = dA_prev_pad[i]
            for h in range(n_H):
                start_H = h*stride
                end_H = h*stride + f_H
                for w in range(n_W):
                    start_W = w*stride
                    end_W = w*stride + f_W
                    for c in range(n_C):

                        A_previ_slice = A_previ_pad[:, start_H:end_H, start_W:end_W]

                        dA_previ_pad[:, start_H:end_H, start_W:end_W] += W[c,:,:,:] * dZ[i, c, h, w]
                        dW[c,:,:,:] += A_previ_slice * dZ[i, c, h, w]
                        db[c,:,:,:] += dZ[i, c, h, w]

            dA_prev[i] = dA_previ_pad[:, pad:-pad, pad:-pad]

        return (dA_prev, dW, db)

    @classmethod
    def __max_mask(cls, X):
        return (X == np.max(X))

    @classmethod
    def __mean_mask(cls, dZ, shape):

        (n_H, n_W) = shape
        avrg = 1/(n_H*n_W)

        return np.one(shape)*avrg

    @classmethod
    def pool_backward(cls, dA, A_prev, f_H, f_W, stride, mode):

        (m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape
        (m, n_C, n_H, n_W) = dA.shape

        dA_prev = np.zeros((m, n_C_prev, n_H_prev, n_W_prev))

        for i in range(m):
            Ai_prev = A_prev[i]
            for h in range(n_H):
                start_H = h*stride
                end_H = h*stride + f_H
                for w in range(n_W):
                    start_W = w*stride
                    end_W = w*stride + f_W
                    for c in range(n_C):

                        if mode == 'max':
                            
                            Ai_prev_slice = Ai_prev[c, start_H:end_H, start_W:end_W]
                            mask = cls.__max_mask(Ai_prev_slice)
                            dA_prev[i,c,start_H:end_H, start_W:end_W] += mask * dA[i,c,start_H:end_H, start_W:end_W]

                        elif mode == 'average':

                            dAi = dA[i]
                            shape = (f_H, f_W)
                            dA_prev[i,c,start_H:end_H, start_W:end_W] += cls.__mean_mask(dAi, shape)

        return dA_prev

                        









