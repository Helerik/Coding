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

    # Initializes Convolutional Neural Network structure
    def __init__(self,
                 layer_sizes = [
                     {'type':'conv', 'f_H':5, 'f_W':5, 'n_C':8, 'stride':1, 'pad':1},
                     {'type':'pool', 'f_H':3, 'f_W':3, 'stride':2, 'mode':'max'},
                     {'type':'fc', 'size':10}
                     ],
                 
                 learning_rate = 0.001,
                 max_iter = 200,
                 L2 = 0, # Not working yet
                 beta1 = 0.9, # Not working yet
                 beta2 = 0.999, # Not working yet
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
            self._activation_str = activation
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

        # fc_weights verifies if fully connected layer weghts have been initialized
        self.fc_weights = False
        self.conv_to_fc_shape = None

        self.weights = {}
        self.best_weights = {}
        self.A_vals = {}
        self.Z_vals = {}
        self.V_vals = {}
        self.S_vals = {}

        self.training_status = "Untrained"

        self.code_breaker = 0

    def __str__(self):

        layer_struct_str = ""
        for i in range(len(self.layer_sizes)):
            if self.layer_sizes[i]['type'] == 'conv':
                layer_struct_str += f"\n    |    Convolutional - Shape:         ({self.layer_sizes[i]['n_C']}, {self.layer_sizes[i]['f_H']}, {self.layer_sizes[i]['f_W']})"
            elif self.layer_sizes[i]['type'] == 'pool':
                layer_struct_str += f"\n    |    {self.layer_sizes[i]['mode'].capitalize()}-Pooling - Shape:           ({self.layer_sizes[i]['f_H']}, {self.layer_sizes[i]['f_W']})"
            elif self.layer_sizes[i]['type'] == 'fc':
                layer_struct_str += f"\n    |    Fully Connected - Size:        {self.layer_sizes[i]['size']}"
        return f"""         {self.classification.capitalize()} Neural Network ({self.training_status}):


    |    Layer Size Structure:{layer_struct_str}
    |
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

    # Plots found out filters
    def plot_filters(self, num_plots):
        for i in range(len(self.best_weights)//2):
            try:
                for j in range(len(self.best_weights['W'+str(i+1)])):
                    for k in range(len(self.best_weights['W'+str(i+1)][j])):
                        plt.figure("Filters Plot")
                        plt.imshow(self.best_weights['W'+str(i+1)][j][k], cmap = 'gray')
                        plt.show()
                        num_plots -= 1
                    if num_plots == 0:
                        return
            except Exception:
                pass    

    # Initializes weights for each layer
    def __initialize_weights(self, n_Cx):

        # Convolutional and pooling weights
        n_C_prev = n_Cx
        for i in range(self.num_layers - 1):

            if self.layer_sizes[i]['type'] == 'conv':

                n_C = self.layer_sizes[i]['n_C']
                f_H = self.layer_sizes[i]['f_H']
                f_W = self.layer_sizes[i]['f_W']
                
                self.weights['W'+str(i+1)] = np.random.randn(n_C, n_C_prev, f_H, f_W)*np.sqrt(2/(f_H*f_W))
                self.weights['b'+str(i+1)] = np.zeros((n_C,1,1,1))

                n_C_prev = n_C

            # There are no pooling layer weights
            elif self.layer_sizes[i]['type'] == 'pool':
                self.weights['W'+str(i+1)] = None
                self.weights['b'+str(i+1)] = None

            # The fully connected weights are added later
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

    # Performs foward propagation
    def __forward_propagation(self):

        Ai_prev = self.minibatch_X
        for i in range(self.num_layers - 1):

            # Performs convolutional layer forward propagation
            if self.layer_sizes[i]['type'] == 'conv':

                stridei = self.layer_sizes[i]['stride']
                padi = self.layer_sizes[i]['pad']
            
                Wi = self.weights['W'+str(i+1)]
                bi = self.weights['b'+str(i+1)]

                Zi = ConvPool.conv_forward(Ai_prev, Wi, bi, stridei, padi)
                Ai = self.activation[i].function(Zi)

                self.A_vals['A'+str(i+1)] = Ai
                self.Z_vals['Z'+str(i+1)] = Zi

                Ai_prev = Ai

            # Performs pooling layer forward propagation
            elif self.layer_sizes[i]['type'] == 'pool':

                f_Hi = self.layer_sizes[i]['f_H']
                f_Wi = self.layer_sizes[i]['f_W']
                stridei = self.layer_sizes[i]['stride']
                modei = self.layer_sizes[i]['mode']

                Ai = ConvPool.pool_forward(Ai_prev, f_Hi, f_Wi, stridei, modei)

                self.A_vals['A'+str(i+1)] = Ai
                self.Z_vals['Z'+str(i+1)] = None

                Ai_prev = Ai

            elif self.layer_sizes[i]['type'] == 'fc':
                k = i
                break

        # Performs 'flattening' on every example of last activation
        Ai = []
        self.conv_to_fc_shape = Ai_prev.shape
        for j in range(Ai_prev.shape[0]):
            Ai.append(Ai_prev[j].flatten())
            
        Ai_prev = np.array(Ai).T
        self.A_vals['A'+str(k)] = Ai_prev

        # Has to initialize weights on fully connected layers if they have not yet been
        # This happens now, so that we can use the shape of the newly made Ai_prev
        if not self.fc_weights:
            self.fc_weights = True
            n_a, m_a = Ai_prev.shape
            n_y, m_y = self.Y.shape
            self.__initialize_fc_weights(n_a, n_y)

        # Performs fully connected layer forward propagation
        for i in range(k, self.num_layers - 1):

            Wi = self.weights['W'+str(i+1)]
            bi = self.weights['b'+str(i+1)]
            
            Zi = np.dot(Wi, Ai_prev) + bi
            Ai = self.activation[i].function(Zi)       

            self.A_vals['A'+str(i+1)] = Ai
            self.Z_vals['Z'+str(i+1)] = Zi
            
            Ai_prev = Ai

        Wi = self.weights['W'+str(self.num_layers)]
        bi = self.weights['b'+str(self.num_layers)]

        # Last layer always receives sigmoid or softmax
        Zi = np.dot(Wi, Ai_prev) + bi
        if self.classification == 'binary':
            Ai = Sigmoid.function(Zi)
        elif self.classification == 'multiclass':
            Ai = Softmax.function(Zi)

        self.A_vals['A'+str(self.num_layers)] = Ai
        self.Z_vals['Z'+str(self.num_layers)] = Zi

    # Performs backward propagation loop
    def __backward_propagation(self):

        # Initialize m
        m = self.minibatch_m
        
        for i in range(self.num_layers, 0, -1):

            # Gets Ai_prev. If i = 1, Ai_prev = X
            if i == 1:
                Ai_prev = self.minibatch_X
            else:
                Ai_prev = self.A_vals['A'+str(i-1)]

            # Gets Zi value
            Zi = self.Z_vals['Z'+str(i)]
            
            # Gets current layer weights
            Wi = self.weights['W'+str(i)]
            bi = self.weights['b'+str(i)]

            # Output layer weight update
            if i == self.num_layers:
                AL = self.A_vals['A'+str(i)]
                dZi = AL - self.minibatch_Y
                
                # Cache dA; on last layer, dA = Wi.T . dZi = Wi.T . (Ai - Y)
                dAi = np.dot(Wi.T, dZi)
                
                # Calculates dWi and dbi
                dWi = np.dot(Ai_prev, dZi.T)/m 
                dbi = np.sum(dZi, axis = 1, keepdims = 1)/m

                # Updates weights and biases
                Wi = Wi - self.learning_rate*dWi.T    
                bi = bi - self.learning_rate*dbi   
                self.weights['W'+str(i)] = Wi
                self.weights['b'+str(i)] = bi

            # Fully connected hidden layer weight update
            elif self.layer_sizes[i-1]['type'] == 'fc':
                dZi = dAi * self.activation[i-1].derivative(Zi)

                # Cache dA; on last layer, dA = Wi.T . dZi = Wi.T . (Ai - Y)
                dAi = np.dot(Wi.T, dZi)

                # Calculates dWi and dbi
                dWi = np.dot(Ai_prev, dZi.T)/m #+ (self.L2/m)*Wi.T
                dbi = np.sum(dZi, axis = 1, keepdims = 1)/m

                # Updates weights and biases
                Wi = Wi - self.learning_rate*dWi.T   
                bi = bi - self.learning_rate*dbi    
                self.weights['W'+str(i)] = Wi
                self.weights['b'+str(i)] = bi

            # Pool layer weight updates
            elif self.layer_sizes[i-1]['type'] == 'pool':

                # If pooling layer was flattened, unflatten it
                if self.layer_sizes[i]['type'] == 'fc':
                    dAi = dAi.reshape(self.conv_to_fc_shape)

                f_Hi = self.layer_sizes[i-1]['f_H']
                f_Wi = self.layer_sizes[i-1]['f_W']
                stridei = self.layer_sizes[i-1]['stride']
                modei = self.layer_sizes[i-1]['mode']

                dAi = ConvPool.pool_backward(dAi, Ai_prev, f_Hi, f_Wi, stridei, modei)

            # Convolutional layer weight updates
            elif self.layer_sizes[i-1]['type'] == 'conv':

                # If convolutional layer was flattened, unflatten it
                if self.layer_sizes[i]['type'] == 'fc':
                    dAi = dAi.reshape(self.conv_to_fc_shape)

                dZi = dAi * self.activation[i-1].derivative(Zi)

                stridei = self.layer_sizes[i-1]['stride']
                padi = self.layer_sizes[i-1]['pad']

                dAi, dWi, dbi = ConvPool.conv_backward(dZi, Ai_prev, Wi, bi, stridei, padi)

                # Updates weights and biases
                Wi = Wi - self.learning_rate*dWi
                bi = bi - self.learning_rate*dbi    
                self.weights['W'+str(i)] = Wi
                self.weights['b'+str(i)] = bi

    # Evaluates cost function
    def __evaluate_cost(self):
        
        AL = self.A_vals['A'+str(self.num_layers)]

        if self.classification == 'binary':
            loss_func = -(self.minibatch_Y*np.log(AL) + (1-self.minibatch_Y)*np.log(1-AL))
        elif self.classification == 'multiclass':
            loss_func = -np.sum(self.minibatch_Y*np.log(AL), axis = 0, keepdims = 1)
        cost_func = np.mean(loss_func)

        # Evaluates regularization cost
##        if self.L2 > 0:
##            L2_reg = 0
##            for i in range(1, self.num_layers):
##                L2_reg += np.sum(np.square(self.weights['W'+str(i)]))
##            L2_reg *= self.L2/(2*self.minibatch_m)
##            cost_func += L2_reg

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

    # Mini-Batch Gradient Descent + ADAM method for training a neural network
    def __mini_batch_gradient_descent(self):
        
        # Cache for plotting cost
        if self.plot_N != None and self.plot_N != 0:
            fig = plt.figure("Cost Plot")

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
        self.best_weights = self.weights
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
                    self.best_weights = self.weights
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
        
        m_x, n_Cx, _, _ = self.X.shape
        n_y, m_y = self.Y.shape
        if m_x != m_y:
            raise ValueError(f"Invalid vector sizes for X and Y -> X size = {X.shape} while Y size = {Y.shape}.")

        if warm_start and self.training_status == "Trained":
            self.weights = self.best_weights
        else:
            self.__initialize_weights(n_Cx)
            
        self.m = m_x

        if self.minibatch_size == None:
            self.minibatch_size = m_x

        # Trains the network
        self.__mini_batch_gradient_descent()
        self.training_status = "Trained"

    # Predicts X vector tags
    def predict(self, X):

        Ai_prev = X
        for i in range(self.num_layers - 1):

            if self.layer_sizes[i]['type'] == 'conv':

                stridei = self.layer_sizes[i]['stride']
                padi = self.layer_sizes[i]['pad']
            
                Wi = self.best_weights['W'+str(i+1)]
                bi = self.best_weights['b'+str(i+1)]

                Zi = ConvPool.conv_forward(Ai_prev, Wi, bi, stridei, padi)
                Ai = self.activation[i].function(Zi)

                Ai_prev = Ai

            elif self.layer_sizes[i]['type'] == 'pool':

                f_Hi = self.layer_sizes[i]['f_H']
                f_Wi = self.layer_sizes[i]['f_W']
                stridei = self.layer_sizes[i]['stride']
                modei = self.layer_sizes[i]['mode']

                Ai = ConvPool.pool_forward(Ai_prev, f_Hi, f_Wi, stridei, modei)

                Ai_prev = Ai

            elif self.layer_sizes[i]['type'] == 'fc':
                k = i
                break

        Ai = []
        self.conv_to_fc_shape = Ai_prev.shape
        for j in range(Ai_prev.shape[0]):
            Ai.append(Ai_prev[j].flatten())
            
        Ai_prev = np.array(Ai).T
        
        for i in range(k, self.num_layers - 1):

            Wi = self.best_weights['W'+str(i+1)]
            bi = self.best_weights['b'+str(i+1)]
            
            Zi = np.dot(Wi, Ai_prev) + bi
            Ai = self.activation[i].function(Zi)       
            
            Ai_prev = Ai

        Wi = self.best_weights['W'+str(self.num_layers)]
        bi = self.best_weights['b'+str(self.num_layers)]

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

# Class for convolution and pooling operations + padding operations
class ConvPool():

    # Pads a matrix with zeros
    @classmethod
    def zero_pad(cls, A_prev, p):
        return np.pad(A_prev, ((0, 0), (0, 0), (p, p), (p, p)), 'constant', constant_values = (0, 0))

    # Performs one convolutional step
    @classmethod
    def __conv_step(cls, A_prev_slice, filt, add):
        return float(np.sum(A_prev_slice * filt) + add)

    # Performs convolution operation between A_prev and the weights, given convolution operation hyperparameters
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

     # Performs pooling operation on A_prev
    @classmethod
    def pool_forward(cls, A_prev, f_H, f_W, stride, mode):

        (m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape

        n_H = int(np.floor((n_H_prev-f_H)/stride + 1))
        n_W = int(np.floor((n_W_prev-f_W)/stride + 1))
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

                        A_slice = A_prev[i, c, start_H:end_H, start_W:end_W]
                        
                        if mode == "max":
                            A[i, c, h, w] = np.max(A_slice)
                        elif mode == "average":
                            A[i, c, h, w] = np.mean(A_slice)

        return A

    # Performs "derivative" of convolution operation between A_prev and the derivative of weights,
    # given convolution operation hyperparameters
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

            if pad != 0:
                dA_prev[i] = dA_previ_pad[:, pad:-pad, pad:-pad]

        return (dA_prev, dW, db)

    # Creates max-pooling backprop mask
    @classmethod
    def __max_mask(cls, X):
        return (X == np.max(X))

    # Creates average-pooling backprop mask
    @classmethod
    def __mean_mask(cls, dZ, shape):

        (n_H, n_W) = shape
        avrg = dZ/(n_H*n_W)

        return np.one(shape)*avrg

    # Performs "derivative" of pooling operation on A_prev, with dA.
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
                            dA_prev[i,c,start_H:end_H, start_W:end_W] += mask * dA[i,c,h,w]

                        elif mode == 'average':

                            dAi = dA[i,c,h,w]
                            shape = (f_H, f_W)
                            dA_prev[i,c,start_H:end_H, start_W:end_W] += cls.__mean_mask(dAi, shape)

        return dA_prev

                        









