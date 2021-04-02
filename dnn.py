import numpy as np
from numpy import random

class NN:
    def __init__(self, dims):
        if len(dims) < 2:
            raise Exception('Wrong dimension: at 2 layers (input and output layers)!')

        self._W = []
        self._b = []
        self._dims = dims

        def random_matrix(row, col):
            w = np.random.randn(row, col)
            w = w / np.linalg.norm(w)
            return w

        for i in range(1, len(dims)):
            lin = dims[i-1]
            lout = dims[i]
            w = random_matrix(lout, lin)
            b = random_matrix(lout, 1)
            self._W.append(w)
            self._b.append(b)

        self._num_layers = len(self._W)

    def _relu(self, a):
        return np.maximum(a, 0)

    def _sigmoid(self, a):
        return 1.0 / (1.0 + np.exp(a))

    def _softmax(self, a):
        ex = np.exp(a)
        return ex / ex.sum()

    def _linear(self, x, w, b):
        return w @ x + b

    def evaluate(self, x):
        if x.shape != (self._dims[0], 1):
            raise Exception('Wrong dimension of input.')

        # forward evaluating the first L-1 layers
        a = x
        for i in range(0, self._num_layers - 1):
            a = self._linear(a, self._W[i], self._b[i])
            a = self._relu(a)

        # evaluate the last layer to the output layer.
        a = self._linear(a, self._W[-1], self._b[-1])
        a = self._softmax(a)

        return a

    def num_input_nodes(self):
        return self._dims[0]

    def num_output_nodes(self):
        return self._dims[-1]
