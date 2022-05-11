import numpy as np
from .function import Function

def _logsumexp(x):
    maxx = x.max(axis=1)
    return maxx + np.log(np.exp(x - maxx.reshape((-1, 1))).sum(axis=1))

class Exp(Function):
    def forward(self, x):
        result = np.exp(x.clip(-50, 50))
        self.save_for_backward(result)
        return result

    def backward(self, grad, **kwargs):
        result, = self.saved_tensors
        return grad * result

class Log(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return np.log(x)

    def backward(self, grad, **kwargs):
        x, = self.saved_tensors
        return grad / x

class ReLU(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return np.maximum(0, x)

    def backward(self, grad, **kwargs):
        x, = self.saved_tensors
        return grad * (x >= 0)

class Sigmoid(Function):
    def forward(self, x):
        result = 1 / (1 + np.exp(-x))
        self.save_for_backward(result)
        return result
    
    def backward(self, grad, **kwargs):
        result, = self.saved_tensors
        return grad * result * (1 - result)

class Tanh(Function):
    """ tldr; this method is extremely slow, look into fast tanh implementations """
    def forward(self, x):
        posexp, negexp = np.exp(x), np.exp(-x)
        result = (posexp - negexp) / (posexp + negexp)
        self.save_for_backward(result)
        return result

    def backward(self, grad, **kwargs):
        result, = self.saved_tensors
        return grad * (1 - result ** 2)

class LogSoftmax(Function):
    """ tldr; might also be slow, investigate faster method """
    def forward(self, x):
        lse = x - _logsumexp(x).reshape((-1, 1))
        self.save_for_backward(lse)
        return lse

    def backward(self, grad, **kwargs):
        # look into derivation of the logsoftmax gradient...
        lse, = self.saved_tensors
        return grad - np.exp(lse) * grad.sum(axis=1).reshape((-1, 1))

