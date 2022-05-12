import numpy as np
from leaf import Tensor

def _tensors_require_grad(*tensors):
    return any(t.requires_grad for t in tensors if isinstance(t, Tensor))

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

    if isinstance(t, (tuple, list)):
        return np.array(t)

    raise ValueError(f'unknown data instance passed as tensor arg to context {context}, {t}')

class Function(object):
    def __init__(self, *tensors):
        self.parents = [t for t in tensors if isinstance(t, Tensor)]
        self.saved_tensors = []
        self.requires_grad = _tensors_require_grad(*tensors)

    def __repr__(self):
        return f'<leaf.ops.{self.__class__.__qualname__}>'

    def save_for_backward(self, *tensors):
        self.saved_tensors.extend(tensors)

    def apply(self, func, *tensors, **kwargs):
        # by registering the ops to the Tensor as an attribute, `self` in
        # this case is the Tensor calling the op, `func` is the 
        # uninitialized operation to perform, e.g. Add, Sub, Mean, Dot etc.
        # `*tensors` are the Tensors to use together with `self` in the `func`,
        # **kwargs are used to specify op specific behavior
        context = func(self, *tensors)
        results = context.forward(self.data,
                *_validate_arg_tensors(context, *tensors), **kwargs)

        if isinstance(results, list):
            results = [Tensor(res, requires_grad=context.requires_grad, _idx=idx,
                _isleaf=False) for idx, res in enumerate(results)]
            for res in results:
                res._ctx = context if res.requires_grad else None

        elif isinstance(results, (np.ndarray, np.float32, np.float64, np.int8, np.int16, np.int32)):
            results = Tensor(results, requires_grad=context.requires_grad, _isleaf=False)
            results._ctx = context if results.requires_grad else None

        return results

