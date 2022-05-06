import numpy as np
from edugrad import Tensor, Function
from functools import partialmethod

def _register(name, func):
    setattr(Tensor, name, partialmethod(func.apply, func))

class Add(Function):
    def forward(self, x, y):
        self.save_for_backward(x.shape, y.shape)
        return x + y

    def backward(self, prev_grad):
        xshape, yshape, = self.saved_tensors
        return np.broadcast_to(prev_grad, xshape), np.broadcast_to(prev_grad, yshape)
_register('add', Add)

class Sub(Function):
    def forward(self, x, y):
        self.save_for_backward(x.shape, y.shape)
        return x - y

    def backward(self, prev_grad):
        xshape, yshape, = self.saved_tensors
        return np.broadcast_to(prev_grad, xshape), -np.broadcast_to(prev_grad, yshape)
_register('sub', Sub)

class Matmul(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x.dot(y)

    def backward(self, prev_grad):
        x, y, = self.saved_tensors
        return prev_grad.dot(y.T), x.T.dot(prev_grad)
_register('matmul', Matmul)

class Mean(Function):
    def forward(self, x, axis=None, keepdims=True):
        out = x.sum(axis=axis, keepdims=keepdims)
        self.save_for_backward(x.shape, out.shape)
        return out * np.prod(out.shape) / np.prod(x.shape)

    def backward(self, prev_grad):
        xshape, outshape, = self.saved_tensors
        return np.ones(xshape) * prev_grad * np.prod(outshape) / np.prod(xshape)
_register('mean', Mean)

class Exp(Function):
    def forward(self, x):
        result = np.exp(x)
        self.save_for_backward(result)
        return result
    
    def backward(self, prev_grad):
        result, = self.saved_tensors
        return prev_grad * result
_register('exp', Exp)

class Log(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return np.log(x)

    def backward(self, prev_grad):
        x, = self.saved_tensors
        return prev_grad / x
_register('log', Log)

class Sum(Function):
    def forward(self, x, axis=None, keepdims=True):
        self.save_for_backward(x.shape)
        return x.sum(axis=axis, keepdims=keepdims)
    
    def backward(self, prev_grad):
        xshape, = self.saved_tensors
        return np.ones(xshape) * prev_grad
_register('sum', Sum)

