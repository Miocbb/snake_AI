import numpy as np


class NN:
    def __init__(self, dims):
        if len(dims) < 2:
            raise Exception(
                'Wrong dimension: at 2 layers (input and output layers)!')

        self.W = []
        self.b = []
        self._dims = dims

        def random_w(row, col):
            w = np.random.uniform(low=-0.5, high=0.5, size=(row, col))
            #w = np.random.randn(row, col)
            #w = w * 0.1
            return w

        def random_b(row, col):
            b = np.random.uniform(low=-0.1, high=0.1, size=(row, col))
            return b

        for i in range(1, len(dims)):
            lin = dims[i-1]
            lout = dims[i]
            w = random_w(lout, lin)
            b = random_b(lout, 1)
            self.W.append(w)
            self.b.append(b)

        self._num_layers = len(self.W)

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
            a = self._linear(a, self.W[i], self.b[i])
            a = self._relu(a)

        # evaluate the last layer to the output layer.
        a = self._linear(a, self.W[-1], self.b[-1])
        #a = self._softmax(a)

        return a

    def num_input_nodes(self):
        return self._dims[0]

    def num_output_nodes(self):
        return self._dims[-1]

    def save(self, path):
        raise Exception('Not implement model save.')

    def load(self, path):
        raise Exception('Not implement model load.')
