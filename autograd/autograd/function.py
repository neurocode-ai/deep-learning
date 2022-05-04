import numpy as np
from .tensor import Tensor
from functools import partialmethod

class Function(object):
    def __init__(self, *tensors):
        self.parents = tensors
        self.saved_tensors = []
        self.requires_grad = any([t.requires_grad for t in tensors if
            isinstance(t, Tensor)])

    def save_for_backward(self, *tensors):
        self.saved_tensors.extend(tensors)

    def apply(self, op, *inputs, **kwargs):
        ctx = op(self, *inputs)
        res = Tensor(ctx.forward(self.data,
            *[t.data for t in inputs], **kwargs))
        res._ctx = ctx
        return res


class Add(Function):
    def forward(self, x, y):
        return x + y
    
    def backward(self, prev_grad):
        return prev_grad, prev_grad
setattr(Tensor, 'add', partialmethod(Add.apply, Add))

class Sum(Function):
    def forward(self, x, axis=None, keepdims=True):
        self.save_for_backward(x.shape)
        return x.sum(axis=axis, keepdims=keepdims)

    def backward(self, prev_grad):
        xshape, = self.saved_tensors
        return prev_grad * np.ones(xshape)
setattr(Tensor, 'sum', partialmethod(Sum.apply, Sum))

