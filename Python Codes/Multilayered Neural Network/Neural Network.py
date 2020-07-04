#!/usr/bin/env python3
# Author: Erik Davino Vincent

import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():

    # Initializes Neural Network structure
    def __init__(self,
                 layer_sizes = [5,5],
                 learning_rate = 0.01,
                 L2 = 0, max_iter = 200,
                 momentum = 0,
                 activation = 'sigmoid',
                 plot_N = None):

        # Structural variables
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) + 1
        self.learning_rate = learning_rate
        self.L2 = L2
        self.momentum = momentum
        self.max_iter = max_iter
        self.plot_N = plot_N

        # Activation function can be string or list
        if isinstance(activation, str):
            if activation.lower() == 'sigmoid':
                self.activation = [Sigmoid for _ in range(self.num_layers-1)]
            elif activation.lower() == 'tanh':
                self.activation = [Tanh for _ in range(self.num_layers-1)]
            elif activation.lower() == 'ltanh':
                self.activation = [LeakyTanh for _ in range(self.num_layers-1)]
            elif activation.lower() == 'relu':
                self.activation = [ReLu for _ in range(self.num_layers-1)]
        else:      
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

        self.weights = {}
        self.best_weights = {}
        self.A_vals = {}
        self.Z_vals = {}
        self.V_vals = {}

    # Initializes momentum for each layer
    def __initialize_momentum(self, n_x, n_y):
        
        n_h_prev = n_x
        for i in range(self.num_layers - 1):
            n_h = self.layer_sizes[i]
            self.V_vals["VdW"+str(i+1)] = np.zeros((n_h, n_h_prev))
            self.V_vals["Vdb"+str(i+1)] = np.zeros((n_h,1))
            n_h_prev = n_h
        
        self.V_vals["VdW"+str(self.num_layers)] = np.zeros((n_y, n_h_prev))
        self.V_vals["Vdb"+str(self.num_layers)] = np.zeros((n_y,1))

    # Initializes weights for each layer
    def __initialize_weights(self, n_x, n_y):
        
        n_h_prev = n_x
        for i in range(self.num_layers - 1):
            n_h = self.layer_sizes[i]
            self.weights['W'+str(i+1)] = np.random.randn(n_h, n_h_prev)*np.sqrt(1/n_h_prev)
            self.weights['b'+str(i+1)] = np.zeros((n_h,1))
            n_h_prev = n_h
        
        self.weights['W'+str(self.num_layers)] = np.random.random((n_y, n_h_prev))*np.sqrt(1/n_h_prev)
        self.weights['b'+str(self.num_layers)] = np.zeros((n_y,1))
    

    # Fits for X and Y
    def fit(self, X, Y, warm_start = False):
        
        n_x, m_x = X.shape
        n_y, m_y = Y.shape
        if m_x != m_y:
            raise ValueError(f"Invalid vector sizes for X and Y -> X size = {X.shape} while Y size = {Y.shape}.")

        if warm_start and self.best_weights:
            self.weights = self.best_weights
        else:
            self.__initialize_weights(n_x, n_y)
        self.__initialize_momentum(n_x, n_y)

        self.X = X
        self.Y = Y
        self.m = m_x

        self.__gradient_descent()

    # Performs foward propagation
    def __forward_propagation(self):

        Ai_prev = np.copy(self.X)
        for i in range(self.num_layers - 1):
            
            Wi = self.weights['W'+str(i+1)]
            bi = self.weights['b'+str(i+1)]
            
            Zi = np.dot(Wi, Ai_prev) + bi
            Ai = self.activation[i].function(Zi)

            self.A_vals['A'+str(i+1)] = Ai
            self.Z_vals['Z'+str(i+1)] = Zi
            
            Ai_prev = np.copy(Ai)

        # Last layer always receives sigmoid
        Wi = self.weights['W'+str(self.num_layers)]
        bi = self.weights['b'+str(self.num_layers)]
        
        Zi = np.dot(Wi, Ai_prev) + bi
        Ai = Sigmoid.function(Zi)

        self.A_vals['A'+str(self.num_layers)] = Ai
        self.Z_vals['Z'+str(self.num_layers)] = Zi

    # Evaluates cost function
    def __evaluate_cost(self):
        
        AL = np.copy(self.A_vals['A'+str(self.num_layers)])
        loss_func = -(self.Y*np.log(AL) + (1-self.Y)*np.log(1-AL))
        cost_func = np.mean(loss_func)

        # Evaluates regularization cost
        if self.L2 > 0:
            L2_reg = 0
            for i in range(1, self.num_layers):
                L2_reg += np.sum(np.square(self.weights['W'+str(i)]))
            L2_reg *= self.L2/(2*self.m)
            cost_func += L2_reg

        return cost_func

    # Newton method for training a neural network
    def __gradient_descent(self):

        # Initialize constant m
        m = self.m
        
        # Cache for ploting cost
        cost = []
        iteration = []

        # Cache for minimum cost and best weights
        self.best_weights = self.weights.copy()
        min_cost = np.inf
        
        for it in range(self.max_iter):

            # Forward propagation
            self.__forward_propagation()

            # Evaluate cost
            cost_func = self.__evaluate_cost()
            
            # Updates best weights
            if cost_func < min_cost:
                self.best_weights = self.weights.copy()
                min_cost = cost_func
                
            # Caches cost function every plot_N iterations
            if self.plot_N != None:
                if (it + 1) % self.plot_N == 0:
                    cost.append(cost_func)
                    iteration.append(it+1)
                    
                    plt.clf()
                    plt.plot(iteration, cost, color = 'b')
                    plt.xlabel("Iteration")
                    plt.ylabel("Cost Function")
                    plt.title(f"Cost Function after {iteration[-1]} iterations:")
                    plt.pause(0.001)

            # Backward propagation
            for i in range(self.num_layers, 0, -1):

                # Gets current layer weights
                Wi = np.copy(self.weights['W'+str(i)])
                bi = np.copy(self.weights['b'+str(i)])
                Ai = np.copy(self.A_vals['A'+str(i)])
                Zi = np.copy(self.Z_vals['Z'+str(i)])

                # Gets momentum
                VdWi = self.V_vals["VdW"+str(i)]
                Vdbi = self.V_vals["Vdb"+str(i)]

                # If on first layer, Ai_prev = X itself
                if i == 1:
                    Ai_prev = np.copy(self.X)
                else:
                    Ai_prev = np.copy(self.A_vals['A'+str(i-1)])

                # If on the last layer, dZi = Ai - Y; else dZi = (Wi+1 . dZi+1) * (Ai*(1-Ai))
                if i == self.num_layers:
                    dZi = Ai - self.Y
                else:
                    dZi = np.dot(Wnxt.T, dZnxt) * self.activation[i-1].derivative(Zi)

                # Calculates dWi and dbi
                dWi = np.dot(Ai_prev, dZi.T)/m + (self.L2/m)*Wi.T
                dbi = np.sum(dZi, axis = 1, keepdims = 1)/m

                # Cache dZi, Wi
                dZnxt = np.copy(dZi)
                Wnxt = np.copy(Wi)

                # Updates momentum
                VdWi = self.momentum*VdWi + (1-self.momentum)*dWi.T
                Vdbi = self.momentum*Vdbi + (1-self.momentum)*dbi

                self.V_vals["VdW"+str(i)] = VdWi
                self.V_vals["Vdb"+str(i)] = Vdbi

                # Updates weights and biases
                Wi = Wi - self.learning_rate*VdWi
                bi = bi - self.learning_rate*Vdbi
                
                self.weights['W'+str(i)] = Wi
                self.weights['b'+str(i)] = bi
                
            # End of backprop loop ==================================================

        if self.plot_N != None:
            
            # Final cost evaluation:
            cost_func = self.__evaluate_cost()
            cost.append(cost_func)
            iteration.append(it+1)
            
            # Plots cost function over time
            plt.clf()
            plt.plot(iteration, cost, color = 'b')
            plt.xlabel("Iteration")
            plt.ylabel("Cost Function")
            plt.title(f"Cost Function after {iteration[-1]} iterations:")
            plt.show(block = 0)

        # Reset variables
        self.X = None
        self.Y = None
        self.m = None

        self.weights = {}
        self.A_vals = {}
        self.Z_vals = {}
        self.V_vals = {}

    # Predicts if X vector tag is 1 or 0
    def predict(self, X):

        Ai_prev = np.copy(X)
        for i in range(self.num_layers - 1):
            
            Wi = self.best_weights['W'+str(i+1)]
            bi = self.best_weights['b'+str(i+1)]
            
            Zi = np.dot(Wi, Ai_prev) + bi
            Ai = Ai = self.activation[i].function(Zi)

            Ai_prev = np.copy(Ai)

        # Last layer always receives sigmoid
        Wi = self.best_weights['W'+str(self.num_layers)]
        bi = self.best_weights['b'+str(self.num_layers)]
        
        Zi = np.dot(Wi, Ai_prev) + bi
        Ai = Sigmoid.function(Zi)

        return Ai > 0.5

# Sigmoid class - contains sigmoid function and its derivative     
class Sigmoid():

    @classmethod
    def function(cls, t):
        return 1/(1+np.exp(-t))

    @classmethod
    def derivative(cls, t):
        return cls.function(t)*(1-cls.function(t))

# Tanh class - contains tanh function and its derivative     
class Tanh():

    @classmethod
    def function(cls, t):
        return np.tanh(t)

    @classmethod
    def derivative(cls, t):
        return 1 - np.power(np.tanh(t), 2)

# "Leaky" Tanh class - contains my idea for a "leaky" tanh function and its derivative
class LeakyTanh():

    @classmethod
    def function(cls, t, leak = 0.2):
        return np.tanh(t) + t*leak

    @classmethod
    def derivative(cls, t, leak = 0.2):
        return 1 - np.power(np.tanh(t), 2) + leak

# ReLu class = contains ReLu function and its derivative
class ReLu():

    @classmethod
    def function(cls, t, leak = 0.0):
        return np.maximum(t*leak, t)

    @classmethod
    def derivative(cls, t, leak = 0.0):
        return np.where(t > 0, 1.0, leak)

def example():

    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler as StdScaler
    from sklearn.model_selection import train_test_split

    # Won't work properly without scalling data
    scaler = StdScaler()
    data = load_breast_cancer(return_X_y = True)

    X = data[0]
    scaler.fit(X)
    X = scaler.transform(X)
    y = data[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    
    X_train = X_train.T
    X_test = X_test.T
    y_train = np.array([y_train])
    y_test = np.array([y_test])

    clf = NeuralNetwork(
        layer_sizes = [20,10],
        learning_rate = 0.1,
        L2 = 0,
        momentum = 0.9,
        max_iter = 500,
        activation = 'tanh',
        plot_N = 20)

    clf.fit(X_train, y_train)
    
    pred = clf.predict(X_train)
    percnt = 0
    for i in range(pred.shape[1]):
        if pred[0,i] == y_train[0,i]:
            percnt += 1
    percnt /= pred.shape[1]
    print()
    print(f"Accuracy of {percnt:.2%} on training set")

    pred = clf.predict(X_test)
    percnt = 0
    for i in range(pred.shape[1]):
        if pred[0,i] == y_test[0,i]:
            percnt += 1
    percnt /= pred.shape[1]
    print()
    print(f"Accuracy of {percnt:.2%} on test set")
    
example()








