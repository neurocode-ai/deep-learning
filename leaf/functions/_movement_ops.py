import numpy as np
from .function import Function

def _slice(arr, arg):
    return arr[tuple([slice(start, stop, None) for start, stop in arg])]

class Reshape(Function):
    def forward(self, x, shape):
        self.save_for_backward(x.shape)
        return x.reshape(shape)

    def backward(self, grad, **kwargs):
        xshape, = self.saved_tensors
        return grad.reshape(xshape)

class Slice(Function):
    """ look into rewrite for this movement op, the backwards pass is hacky """
    def forward(self, x, arg=None, nshape=None):
        self.save_for_backward(x.shape, arg, nshape)
        return _slice(x, arg).reshape(nshape)

    def backward(self, grad, **kwargs):
        xshape, arg, nshape, = self.saved_tensors
        result = np.zeros(xshape)
        rshape = result[tuple([slice(start, stop, None) for start, stop in arg])].shape
        result[tuple([slice(start, stop, None) for start, stop in arg])] = grad.reshape(rshape)
        return result

class Chunk(Function):
    def forward(self, x, chunks=1, dim=0):
        result = np.split(x, chunks, dim)
        self.save_for_backward(x.shape, chunks, dim)
        return result

    def backward(self, grad, _idx=None):
        xshape, chunks, dim, = self.saved_tensors
        struct = [np.zeros_like(grad) if i != _idx else grad for i in range(chunks)]
        return np.concatenate(struct, axis=dim).reshape(xshape)

class Transpose(Function):
    def forward(self, x, order):
        self.save_for_backward(order)
        return np.transpose(x, order)

    def backward(self, grad, **kwargs):
        order, = self.saved_tensors
        return np.transpose(grad, tuple(np.argsort(order)))

class Concatenate(Function):
    def forward(self, _, *tensors, dim=0):
        self.save_for_backward(dim, [t.shape for t in tensors])
        return np.concatenate([np.expand_dims(t.data, dim) for t in tensors], axis=dim)

    def backward(self, grad, **kwargs):
        dim, shapes, = self.saved_tensors
        outs = np.split(grad, len(shapes), dim)
        for i, (out, shape) in enumerate(zip(outs, shapes)):
            outs[i] = out.reshape(shape)
        outs.insert(0, None)
        return outs

