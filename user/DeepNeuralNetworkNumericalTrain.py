import numpy as np

def _m(x, y)
    return np.Matmul(x, y)

def _t(x)
    return np.Transpose(x)

class Neuron:
    def __init__(self, W, b, a):
        self.W = W
        self.b = b
        self.a = a

        self.dW = np.zero_like(W)
        self.db = np.zero_like(b)

    def __call__(self, x):
        self.a(_m(self.W, x) + self.b)