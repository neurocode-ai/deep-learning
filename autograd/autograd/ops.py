import numpy as np
from autograd import Function
from autograd import Tensor
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

class Mean(Function):
    def forward(self, x, axis=None, keepdims=True):
        self.save_for_backward(x.shape)
        return x.sum(axis=axis, keepdims=keepdims) / np.prod(x.shape)

    def backward(self, prev_grad):
        xshape, = self.saved_tensors
        return np.ones(xshape) * prev_grad / np.prod(xshape)
_register('mean', Mean)

