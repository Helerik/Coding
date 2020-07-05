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
                 beta1 = 0,
                 beta2 = 0,
                 activation = 'sigmoid',
                 epsilon = 1e-8,
                 minibatch_size = None,
                 plot_N = None):

        # Structural variables
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) + 1
        self.learning_rate = learning_rate
        self.L2 = L2
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.minibatch_size = minibatch_size
        self.max_iter = max_iter
        self.plot_N = plot_N

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

        self.weights = {}
        self.best_weights = {}
        self.A_vals = {}
        self.Z_vals = {}
        self.V_vals = {}
        self.S_vals = {}

        self.training_status = "Untrained"

    def __str__(self):
        return f"""Neural Network ({self.training_status}):

Layer Size Structure:          {np.array(self.layer_sizes)}
Activation Function Structure: {self._activation_str}

Learning Rate:                 {self.learning_rate}
L2 Regularization:             {self.L2}

Beta1 (Momentum):              {self.beta1}
Beta2 (RMSprop):               {self.beta2}
Epsilon:                       {self.epsilon:.1e}

Mini-Batch Size:               {self.minibatch_size}
Max Iterations:                {self.max_iter}"""

    # Initializes momentum for each layer
    def __initialize_momentum(self, n_x, n_y):
        
        n_h_prev = n_x
        for i in range(self.num_layers - 1):
            n_h = self.layer_sizes[i]
            
            self.V_vals["VdW"+str(i+1)] = np.zeros((n_h, n_h_prev))
            self.V_vals["Vdb"+str(i+1)] = np.zeros((n_h,1))
            
            self.S_vals["SdW"+str(i+1)] = np.zeros((n_h, n_h_prev))
            self.S_vals["Sdb"+str(i+1)] = np.zeros((n_h,1))
            
            n_h_prev = n_h
        
        self.V_vals["VdW"+str(self.num_layers)] = np.zeros((n_y, n_h_prev))
        self.V_vals["Vdb"+str(self.num_layers)] = np.zeros((n_y,1))
        
        self.S_vals["SdW"+str(self.num_layers)] = np.zeros((n_y, n_h_prev))
        self.S_vals["Sdb"+str(self.num_layers)] = np.zeros((n_y,1))      

    # Initializes weights for each layer
    def __initialize_weights(self, n_x, n_y):
        
        n_h_prev = n_x
        for i in range(self.num_layers - 1):
            n_h = self.layer_sizes[i]
            self.weights['W'+str(i+1)] = np.random.randn(n_h, n_h_prev)*np.sqrt(2/n_h_prev)
            self.weights['b'+str(i+1)] = np.zeros((n_h,1))
            n_h_prev = n_h
        
        self.weights['W'+str(self.num_layers)] = np.random.random((n_y, n_h_prev))*np.sqrt(2/n_h_prev)
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

        if self.minibatch_size == None:
            self.minibatch_size = m_x

        self.__mini_batch_gradient_descent()
        self.training_status = "Trained"

    # Performs foward propagation
    def __forward_propagation(self):

        Ai_prev = np.copy(self.minibatch_X)
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
        loss_func = -(self.minibatch_Y*np.log(AL) + (1-self.minibatch_Y)*np.log(1-AL))
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

        # Initializes minibatches list
        self.minibatches = []

        # Shuffles data before creating minibatches
        permutation = list(np.random.permutation(self.m))
        shuffled_X = self.X[:, permutation]
        shuffled_Y = self.Y[:, permutation].reshape((1,self.m))

        num_full_minibatches = int(np.floor(self.m/self.minibatch_size))
        for k in range(num_full_minibatches):
            minibatch_X = shuffled_X[:, self.minibatch_size*k : self.minibatch_size*(k+1)]
            minibatch_Y = shuffled_Y[:, self.minibatch_size*k : self.minibatch_size*(k+1)]
            minibatch = (minibatch_X, minibatch_Y)
            self.minibatches.append(minibatch)

        # Depending on size of X and size of minibatches, the last minibatch will be smaller
        if self.m % self.minibatch_size != 0:
            minibatch_X = shuffled_X[:, self.minibatch_size*num_full_minibatches :]
            minibatch_Y = shuffled_Y[:, self.minibatch_size*num_full_minibatches :]
            minibatch = (minibatch_X, minibatch_Y)
            self.minibatches.append(minibatch)  

    # Newton method for training a neural network
    def __mini_batch_gradient_descent(self):
        
        # Cache for plotting cost
        cost = []
        iteration = []

        # Cache for minimum cost and best weights
        self.best_weights = self.weights.copy()
        min_cost = np.inf

        for it in range(self.max_iter):

            # Creates iteration minibatches
            self.__make_minibatches()

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
                
                # Updates best weights
                if self.best_minibatch_cost < min_cost:
                    self.best_weights = self.weights.copy()
                    min_cost = self.best_minibatch_cost

                # Backward propagation
                for i in range(self.num_layers, 0, -1):

                    # Gets current layer weights
                    Wi = np.copy(self.weights['W'+str(i)])
                    bi = np.copy(self.weights['b'+str(i)])
                    Ai = np.copy(self.A_vals['A'+str(i)])
                    Zi = np.copy(self.Z_vals['Z'+str(i)])

                    # Gets momentum
                    VdWi = np.copy(self.V_vals["VdW"+str(i)])
                    Vdbi = np.copy(self.V_vals["Vdb"+str(i)])

                    # Gets RMSprop
                    SdWi = np.copy(self.S_vals["SdW"+str(i)])
                    Sdbi = np.copy(self.S_vals["Sdb"+str(i)])

                    # If on first layer, Ai_prev = X itself
                    if i == 1:
                        Ai_prev = np.copy(self.minibatch_X)
                    else:
                        Ai_prev = np.copy(self.A_vals['A'+str(i-1)])

                    # If on the last layer, dZi = Ai - Y; else dZi = (Wi+1 . dZi+1) * (Ai*(1-Ai))
                    if i == self.num_layers:
                        dZi = Ai - self.minibatch_Y
                    else:
                        dZi = np.dot(Wnxt.T, dZnxt) * self.activation[i-1].derivative(Zi)

                    # Calculates dWi and dbi
                    dWi = np.dot(Ai_prev, dZi.T)/m + (self.L2/m)*Wi.T
                    dbi = np.sum(dZi, axis = 1, keepdims = 1)/m

                    # Cache dZi, Wi
                    dZnxt = np.copy(dZi)
                    Wnxt = np.copy(Wi)

                    # Updates momentum
                    VdWi = self.beta1*VdWi + (1-self.beta1)*dWi.T
                    Vdbi = self.beta1*Vdbi + (1-self.beta1)*dbi

                    self.V_vals["VdW"+str(i)] = VdWi
                    self.V_vals["Vdb"+str(i)] = Vdbi

                    # Updates RMSprop
                    SdWi = self.beta2*SdWi + (1-self.beta2)*np.square(dWi.T)
                    Sdbi = self.beta2*Sdbi + (1-self.beta2)*np.square(dbi)

                    self.S_vals["SdW"+str(i)] = SdWi
                    self.S_vals["Sdb"+str(i)] = Sdbi

                    # Updates weights and biases
                    Wi = Wi - self.learning_rate*VdWi/(np.sqrt(SdWi) + self.epsilon)
                    bi = bi - self.learning_rate*Vdbi/(np.sqrt(Sdbi) + self.epsilon)

                    self.weights['W'+str(i)] = Wi
                    self.weights['b'+str(i)] = bi
                    
                # End of backprop loop ==================================================
                
            # Caches cost function every plot_N iterations
            if self.plot_N != None:
                if (it + 1) % self.plot_N == 0:
                    cost.append(self.best_minibatch_cost)
                    iteration.append(it+1)
                    plt.clf()
                    plt.plot(iteration, cost, color = 'b')
                    plt.xlabel("Iteration")
                    plt.ylabel("Cost Function")
                    plt.title(f"Cost Function over {iteration[-1]} iterations:")
                    plt.pause(0.001)

        # Final plot
        if self.plot_N != None:
            
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
    def function(cls, t, leak = 0.1):
        return np.maximum(t, leak)

    @classmethod
    def derivative(cls, t, leak = 0.1):
        dt = np.copy(t)
        dt[t <= 0] = leak
        dt[t > 0] = 1
        return dt

# Softmax class - contains softmax function
class Softmax():

    @classmethod
    def function(cls, t):
        denominator = np.sum(np.exp(t))
        return t/denominator

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
        layer_sizes = [10, 10],
        learning_rate = 0.01,
        L2 = 0,
        beta1 = 0.9,
        beta2 = 0.999,
        max_iter = 100,
        minibatch_size = 128,
        activation = ['sigmoid', 'ltanh'],
        plot_N = 1)

    print()
    print(clf)

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








