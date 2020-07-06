#!/usr/bin/env python3
# Author: Erik Davino Vincent

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork():

    # Initializes Neural Network structure
    def __init__(self,
                 layer_sizes = [5,5],
                 learning_rate = 0.01,
                 max_iter = 200,
                 L2 = 0,
                 beta1 = 0,
                 beta2 = 0.999,
                 activation = 'sigmoid',
                 epsilon = 1e-8,
                 minibatch_size = None,
                 classification = 'binary',
                 plot_N = None):

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
        return f"""{self.classification.capitalize()} Neural Network ({self.training_status}):

Layer Size Structure:          {np.array(self.layer_sizes)}
Activation Function Structure: {self._activation_str}

Learning Rate:                 {self.learning_rate}
L2 Regularization:             {self.L2}

Beta1 (Momentum):              {self.beta1}
Beta2 (RMSprop):               {self.beta2}
Epsilon:                       {self.epsilon:.1e}

Mini-Batch Size:               {self.minibatch_size}
Max Iterations:                {self.max_iter}"""

    # Initializes momentum and RMSprop for each layer
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

        self.weights['W'+str(self.num_layers)] = np.random.randn(n_y, n_h_prev)*np.sqrt(2/n_h_prev)
        self.weights['b'+str(self.num_layers)] = np.zeros((n_y,1))

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

    # Performs foward propagation
    def __forward_propagation(self):

        Ai_prev = np.copy(self.minibatch_X)
        for i in range(self.num_layers - 1):
            
            Wi = np.copy(self.weights['W'+str(i+1)])
            bi = np.copy(self.weights['b'+str(i+1)])
            
            Zi = np.dot(Wi, Ai_prev) + bi
            Ai = self.activation[i].function(Zi)       

            self.A_vals['A'+str(i+1)] = np.copy(Ai)
            self.Z_vals['Z'+str(i+1)] = np.copy(Zi)
            
            Ai_prev = np.copy(Ai)

        Wi = np.copy(self.weights['W'+str(self.num_layers)])
        bi = np.copy(self.weights['b'+str(self.num_layers)])

        # Last layer always receives sigmoid or softmax
        Zi = np.dot(Wi, Ai_prev) + bi
        if self.classification == 'binary':
            Ai = Sigmoid.function(Zi)
        elif self.classification == 'multiclass':
            Ai = Softmax.function(Zi)

        self.A_vals['A'+str(self.num_layers)] = np.copy(Ai)
        self.Z_vals['Z'+str(self.num_layers)] = np.copy(Zi)

    # Evaluates cost function
    def __evaluate_cost(self):
        
        AL = np.copy(self.A_vals['A'+str(self.num_layers)])

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
        shuffled_X = self.X[:, permutation]
        shuffled_Y = self.Y[:, permutation]

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
        if self.plot_N != None and self.plot_N != 0:
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

                    # If on the last layer, dZi = Ai - Y; else dZi = (Wi+1 . dZi+1) * g'(Zi)
                    if i == self.num_layers:
                        dZi = Ai - self.minibatch_Y
                    else:
                        dZi = np.dot(Wnxt.T, dZnxt) * self.activation[i-1].derivative(Zi)

                    # Cache dZi, Wi
                    dZnxt = np.copy(dZi)
                    Wnxt = np.copy(Wi)

                    # Calculates dWi and dbi
                    dWi = np.dot(Ai_prev, dZi.T)/m + (self.L2/m)*Wi.T
                    dbi = np.sum(dZi, axis = 1, keepdims = 1)/m

                    # Updates momentum
                    VdWi = self.beta1*VdWi + (1-self.beta1)*dWi.T
                    Vdbi = self.beta1*Vdbi + (1-self.beta1)*dbi
                    self.V_vals["VdW"+str(i)] = np.copy(VdWi)
                    self.V_vals["Vdb"+str(i)] = np.copy(Vdbi)

                    # Updates RMSprop
                    SdWi = self.beta2*SdWi + (1-self.beta2)*np.square(dWi.T)
                    Sdbi = self.beta2*Sdbi + (1-self.beta2)*np.square(dbi)
                    self.S_vals["SdW"+str(i)] = np.copy(SdWi)
                    self.S_vals["Sdb"+str(i)] = np.copy(Sdbi)

                    # Updates weights and biases
                    Wi = Wi - self.learning_rate*VdWi/(np.sqrt(SdWi) + self.epsilon)
                    bi = bi - self.learning_rate*Vdbi/(np.sqrt(Sdbi) + self.epsilon)
                    self.weights['W'+str(i)] = np.copy(Wi)
                    self.weights['b'+str(i)] = np.copy(bi)
                    
                # End of backprop loop ==================================================
                
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

    # Predicts X vector tags
    def predict(self, X):

        Ai_prev = np.copy(X)
        for i in range(self.num_layers - 1):
            
            Wi = np.copy(self.best_weights['W'+str(i+1)])
            bi = np.copy(self.best_weights['b'+str(i+1)])
            
            Zi = np.dot(Wi, Ai_prev) + bi
            Ai = self.activation[i].function(Zi)

            Ai_prev = np.copy(Ai)

        Wi = np.copy(self.best_weights['W'+str(self.num_layers)])
        bi = np.copy(self.best_weights['b'+str(self.num_layers)])

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

# ReLu class - contains ReLu function and its derivative
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
        new_t = np.exp(t)
        denominator = np.sum(new_t, axis = 0, keepdims = 1)
        return np.divide(new_t, denominator)

def example():

    from sklearn.datasets import load_breast_cancer, load_iris
    from sklearn.preprocessing import StandardScaler as StdScaler
    from sklearn.model_selection import train_test_split

    # Won't work properly without scalling data
    scaler = StdScaler()

    # Importing data from sklearn datasets
##    data = load_breast_cancer(return_X_y = True)
    data = load_iris(return_X_y = True)

    # Scalling X
    X = data[0]
    scaler.fit(X)
    X = scaler.transform(X)
    y = data[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    
    X_train = X_train.T
    X_test = X_test.T
    y_train = np.array([y_train])
    y_test = np.array([y_test])

    # Initializes NN classifier
    clf = NeuralNetwork(
        layer_sizes = [10],
        learning_rate = 0.001,
        max_iter = 500,
        L2 = 0,
        beta1 = 0.9,
        beta2 = 0.999,
        minibatch_size = None,
        activation = 'tanh',
        classification = 'multiclass',
        plot_N = 50)

    print()
    print(clf)

    clf.fit(X_train, y_train)

    # Makes predictions
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








