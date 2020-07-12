#!/usr/bin/env python3
# Author: Erik Davino Vincent

import numpy as np

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
    def function(cls, t, leak = 0.01):
        return np.tanh(t) + t*leak

    @classmethod
    def derivative(cls, t, leak = 0.01):
        return 1 - np.power(np.tanh(t), 2) + leak

# ReLu class - contains (leaky) ReLu function and its derivative
class ReLu():

    @classmethod
    def function(cls, t, leak = 0.01):
        return np.maximum(t, t*leak)

    @classmethod
    def derivative(cls, t, leak = 0.01):
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
