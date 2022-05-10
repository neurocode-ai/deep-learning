import numpy as np
from leaf import Tensor, Function
from functools import partialmethod

def _register(name, func):
    setattr(Tensor, name, partialmethod(func.apply, func))

def _unbroadcast(arr, shape):
    return np.lib.stride_tricks.as_strided(arr, shape).copy()

class Add(Function):
    def forward(self, x, y):
        self.save_for_backward(x.shape, y.shape)
        return x + y

    def backward(self, prev_grad, **kwargs):
        xshape, yshape, = self.saved_tensors
        return _unbroadcast(prev_grad, xshape), _unbroadcast(prev_grad, yshape)
_register('add', Add)

class Sub(Function):
    def forward(self, x, y):
        self.save_for_backward(x.shape, y.shape)
        return x - y

    def backward(self, prev_grad, **kwargs):
        xshape, yshape, = self.saved_tensors
        return _unbroadcast(prev_grad, xshape), - _unbroadcast(prev_grad, yshape)
_register('sub', Sub)

class Matmul(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x @ y

    def backward(self, prev_grad, **kwargs):
        x, y, = self.saved_tensors
        return prev_grad @ y.T, x.T @ prev_grad
_register('matmul', Matmul)

class Multiply(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return np.multiply(x, y)

    def backward(self, prev_grad, **kwargs):
        x, y, = self.saved_tensors
        return _unbroadcast(prev_grad * y, x.shape), _unbroadcast(prev_grad * x, y.shape)
_register('multiply', Multiply)

class Mean(Function):
    def forward(self, x, axis=None, keepdims=True):
        out = x.sum(axis=axis, keepdims=keepdims)
        self.save_for_backward(x.shape, out.shape)
        return out * np.prod(out.shape) / np.prod(x.shape)

    def backward(self, prev_grad, **kwargs):
        xshape, outshape, = self.saved_tensors
        return np.ones(xshape) * prev_grad * np.prod(outshape) / np.prod(xshape)
_register('mean', Mean)

class Exp(Function):
    def forward(self, x):
        result = np.exp(x)
        self.save_for_backward(result)
        return result
    
    def backward(self, prev_grad, **kwargs):
        result, = self.saved_tensors
        return prev_grad * result
_register('exp', Exp)

class Log(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return np.log(x)

    def backward(self, prev_grad, **kwargs):
        x, = self.saved_tensors
        return prev_grad / x
_register('log', Log)

class Sum(Function):
    def forward(self, x, axis=None, keepdims=True):
        self.save_for_backward(x.shape)
        return x.sum(axis=axis, keepdims=keepdims)
    
    def backward(self, prev_grad, **kwargs):
        xshape, = self.saved_tensors
        return np.ones(xshape) * prev_grad
_register('sum', Sum)

class Sigmoid(Function):
    def forward(self, x):
        result = 1 / (1 + np.exp(-x))
        self.save_for_backward(result)
        return result
    
    def backward(self, prev_grad, **kwargs):
        result, = self.saved_tensors
        return prev_grad * result * (1 - result)
_register('sigmoid', Sigmoid)

class ReLU(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return np.maximum(x, 0)

    def backward(self, prev_grad, **kwargs):
        x, = self.saved_tensors
        return prev_grad * (x >= 0)
_register('relu', ReLU)

class Tanh(Function):
    def forward(self, x):
        posexp, negexp = np.exp(x), np.exp(-x)
        result = (posexp - negexp) / (posexp + negexp)
        self.save_for_backward(result)
        return result

    def backward(self, prev_grad, **kwargs):
        result, = self.saved_tensors
        return prev_grad * (1 - np.power(result, 2))
_register('tanh', Tanh)

class Pow(Function):
    def forward(self, x, y):
        result = np.power(x, y)
        self.save_for_backward(x, y, result)
        return result

    def backward(self, prev_grad, **kwargs):
        x, y, power, = self.saved_tensors
        return _unbroadcast(prev_grad * y * np.power(x, y - 1), x.shape), \
                _unbroadcast(prev_grad * power * np.log(x), y.shape)
_register('pow', Pow)

class Reshape(Function):
    def forward(self, x, shape):
        self.save_for_backward(x.shape)
        return x.reshape(shape)

    def backward(self, prev_grad, **kwargs):
        xshape, = self.saved_tensors
        return prev_grad.reshape(xshape)
_register('reshape', Reshape)

class Chunk(Function):
    def forward(self, x, chunks=1, dim=0):
        result = np.split(x, chunks, dim)
        self.save_for_backward(x.shape, chunks, dim)
        return result

    def backward(self, prev_grad, _idx=None):
        xshape, chunks, dim, = self.saved_tensors
        remake = [np.zeros_like(prev_grad) if i != _idx else prev_grad for i in range(chunks)]
        result = np.concatenate(remake, axis=dim).reshape(xshape)
        return result
_register('chunk', Chunk)

class LogSoftmax(Function):
    def forward(self, x):
        def _logsumexp(y):
            m = y.max(axis=1)
            return m + np.log(np.exp(y - m.reshape((-1, 1))).sum(axis=1))
        lse = x - _logsumexp(x).reshape((-1, 1))
        self.save_for_backward(lse)
        return lse
    
    def backward(self, prev_grad, **kwargs):
        lse, = self.saved_tensors
        return prev_grad - np.exp(lse) * prev_grad.sum(axis=1).reshape((-1, 1))
_register('logsoftmax', LogSoftmax)

