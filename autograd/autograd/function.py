import numpy as np
from autograd import Tensor

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

