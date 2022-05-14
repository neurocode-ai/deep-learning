import numpy as np
from .function import Function

class Mean(Function):
    def forward(self, x, axis=None, keepdims=True):
        res = x.sum(axis=axis, keepdims=keepdims)
        self.save_for_backward(x.shape, res.shape)
        return res * np.prod(res.shape) / np.prod(x.shape)
    
    def backward(self, grad, **kwargs):
        xshape, rshape, = self.saved_tensors
        return np.ones(xshape) * grad * np.prod(rshape) / np.prod(xshape)

class Sum(Function):
    def forward(self, x, axis=None, keepdims=True):
        self.save_for_backward(x.shape)
        return x.sum(axis=axis, keepdims=keepdims)

    def backward(self, grad, **kwargs):
        xshape, = self.saved_tensors
        return np.ones(xshape) * grad

class Max(Function):
    def forward(self, x, axis=None, keepdims=True):
        res = np.amax(x, axis=axis, keepdims=keepdims)
        self.save_for_backward(x, axis, keepdims, res)
        return res

    def backward(self, grad, **kwargs):
        x, axis, keepdims, res, = self.saved_tensors
        mask = (x == res)
        return mask * grad / mask.sum(axis=axis, keepdims=keepdims)

