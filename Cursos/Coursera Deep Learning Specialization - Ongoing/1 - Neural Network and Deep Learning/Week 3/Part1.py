
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def tanhyp(z):
    return np.tanh(z)

def initializeWeights(n_x, n_h, n_y, seed = None):
    np.random.seed(seed)
    
    W1 = np.random.random((n_h, n_x))/10
    b1 = np.zeros((n_h,1))
    W2 = np.random.random((n_y, n_h))/10
    b2 = np.zeros((n_y,1))

    weights = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
        }
    
    np.random.seed(None)
    return weights

def activationGrad(A, activationFunction):

    if activationFunction == "tanh":
        return (1 - np.power(A, 2))
    elif activationFunction == "sigmoid":
        return A * (1 - A)

def gradientDescent_NN(X, Y, weights, alpha, max_iter, activationFunction = "tanh"):

    if activationFunction == "tanh":
        actFun = tanhyp
    elif activationFunction == "sigmoid":
        actFun = sigmoid

    W1 = weights['W1']
    b1 = weights['b1']
    W2 = weights['W2']
    b2 = weights['b2']

    cost = []
    
    m = X.shape[1]
    for _ in range(max_iter):

        # Forward propagation
        Z1 = np.dot(W1, X) + b1
        A1 = actFun(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)

        lossFunc = -(Y*np.log(A2) + (1-Y)*np.log(1-A2))
        costFunc = np.sum(lossFunc)/m
        cost.append(costFunc)

        # Backward propagation
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T)/m
        db2 = np.sum(dZ2, axis = 1, keepdims = 1)/m
        
        dZ1 = np.dot(W2.T, dZ2) * activationGrad(A1, activationFunction)
        dW1 = np.dot(dZ1, X.T)/m
        db1 = np.sum(dZ1, axis = 1, keepdims = 1)/m

        W1 = W1 - alpha*dW1
        b1 = b1 - alpha*db1
        W2 = W2 - alpha*dW2
        b2 = b2 - alpha*db2
    
    weights = {
    "W1": W1,
    "b1": b1,
    "W2": W2,
    "b2": b2
    }

    plt.plot(cost, color = 'b')
    plt.title("Cost function")
    plt.show(block = 0)

    return weights

def predict(weights, X, activationFunction = "tanh"):

    if activationFunction == "tanh":
        actFun = tanhyp
    elif activationFunction == "sigmoid":
        actFun = sigmoid

    W1 = weights['W1']
    b1 = weights['b1']
    W2 = weights['W2']
    b2 = weights['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = actFun(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    return (A2 > 0.5)

def model(X, Y, layerSize, alpha, max_iter = 100, activationFunction = "tanh"):

    n_x, m = X.shape
    n_h = layerSize
    n_y = Y.shape[0]
    if m != Y.shape[1]:
        raise ValueError("Invalid vector sizes for trainX and trainY -> trainX size = " + str(self.m) + " while trainY size = " + str(len(Y)))

    weights = initializeWeights(n_x, n_h, n_y)

    return gradientDescent_NN(X, Y, weights, alpha, max_iter, activationFunction)

def example():
    
    X = np.array([
        [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
        [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
        [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
        ])
    y = np.array([[1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]])
    X[0] = X[0]/100

    weights = model(X, y, 10, 0.075, 4000)

    pred = predict(weights, X)

    percnt = 0
    for i in range(pred.shape[1]):
        if pred[0,i] == y[0,i]:
            percnt += 1
    percnt /= pred.shape[1]

    print("Accuracy of", percnt*100 , "%")

example()



    
