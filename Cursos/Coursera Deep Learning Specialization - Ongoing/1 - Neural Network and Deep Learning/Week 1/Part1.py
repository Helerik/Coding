
import numpy as np

def sigmoid(t):
    return 1/(1+np.exp(-t))

def logisticRegression(X, w, b):
    return sigmoid(np.dot(w.T, x) + b)
    
