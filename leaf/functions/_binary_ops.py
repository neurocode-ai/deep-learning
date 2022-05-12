import numpy as np
from .function import Function

def _unbroadcast(arr, shape):
    return np.lib.stride_tricks.as_strided(arr, shape).copy()

class Add(Function):
    def forward(self, x, y):
        self.save_for_backward(x.shape, y.shape)
        return x + y
    
    def backward(self, grad, **kwargs):
        xshape, yshape, = self.saved_tensors
        return _unbroadcast(grad, xshape), _unbroadcast(grad, yshape)

class Sub(Function):
    def forward(self, x, y):
        self.save_for_backward(x.shape, y.shape)
        return x - y

    def backward(self, grad, **kwargs):
        xshape, yshape, = self.saved_tensors
        return _unbroadcast(grad, xshape), - _unbroadcast(grad, yshape)

class Mul(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x * y

    def backward(self, grad, **kwargs):
        x, y, = self.saved_tensors
        return _unbroadcast(y * grad, x.shape), _unbroadcast(x * grad, y.shape)

class Multiply(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return np.multiply(x, y)

    def backward(self, grad, **kwargs):
        x, y, = self.saved_tensors
        return _unbroadcast(y * grad, x.shape), _unbroadcast(x * grad, y.shape)

class Pow(Function):
    def forward(self, x, y):
        result = x ** y
        self.save_for_backward(x, y, result)
        return result

    def backward(self, grad, **kwargs):
        x, y, result, = self.saved_tensors
        logged = np.log(x, out=np.zeros_like(x), where=(x>0))
        return _unbroadcast(y * grad * x ** (y - 1), x.shape), \
                _unbroadcast(result * grad * logged, y.shape)

