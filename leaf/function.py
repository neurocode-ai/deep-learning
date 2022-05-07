import numpy as np
from leaf import Tensor

class Function(object):
    def __init__(self, *tensors):
        self.parents = tensors
        self.saved_tensors = []
        self.requires_grad = _tensors_require_grad(*tensors)

    def save_for_backward(self, *tensors):
        self.saved_tensors.extend(tensors)

    def apply(self, func, *tensors, **kwargs):
        # by registering the ops to the Tensor as an attribute, `self` in
        # this case is the Tensor calling the op, `func` is the 
        # uninitialized operation to perform, e.g. Add, Sub, Mean, Dot etc.
        # `*tensors` are the Tensors to use together with `self` in the `func`,
        # **kwargs are used to specify op specific behavior
        context = func(self, *tensors)
        result = Tensor(context.forward(self.data,
            *_validate_arg_tensors(context, *tensors),
            **kwargs), requires_grad=context.requires_grad)
        result._ctx = context
        return result

def _validate_arg_tensors(context, *tensors):
    return [_extract_data(context, t) for t in tensors]

def _extract_data(context, t):
    if isinstance(t, np.ndarray):
        return t

    if isinstance(t, Tensor):
        return t.data

    if isinstance(t, int):
        return np.array([t]).astype(int)

    if isinstance(t, float):
        return np.array([t]).astype(float)

    if isinstance(t, tuple) or isinstance(t, list):
        return np.array(t)

    raise ValueError(f'unknown data instance passed as tensor arg to context' + 
            f'{context}, {t}')

def _tensors_require_grad(*tensors):
    return any(t.requires_grad for t in tensors if isinstance(t, Tensor))

