import numpy as np

class Tensor(object):
    def __init__(self, data, requires_grad=False, dtype=np.float32, _idx=None, _isleaf=True):
        self.requires_grad = requires_grad

        if isinstance(data, (int, float)):
            data = np.array([data]).astype(dtype)

        elif isinstance(data, (tuple, list)):
            data = np.array(data).astype(dtype)

        elif isinstance(data, (np.float32, np.float64)):
            data = np.array([data]).astype(dtype)

        elif isinstance(data, np.ndarray):
            data = data.astype(dtype)

        else: raise ValueError(
        f'unknown data instance passed to Tensor, {type(data)}')

        self.data = data
        self.grad = None
        self._ctx = None
        self._idx = _idx
        self._isleaf = _isleaf

    def __str__(self):
        return f'<leaf.Tensor(\n{self.data}\n' \
                'dtype={self.dtype}, grad_fn={self._ctx}, grad={self.grad}>'

    def __getitem__(self, arg):
        indices = []
        nshape = []

        if isinstance(arg, (int, float, tuple, slice)):
            for i, s in enumerate(arg if isinstance(arg, (list, tuple)) else [arg]):
                if isinstance(s, int):
                    # arg is either an int, list, or tuple
                    indices.append((s, s + 1))
                    continue

                # arg is slice
                indices.append((s.start if s.start is not None else 0,
                    (s.stop if s.stop >= 0 else self.shape[i]+s.stop) if s.stop is not None else self.shape[i]))
                nshape.append(indices[-1][1] - indices[-1][0])
                assert s.step is None or s.step == 1
            nshape += self.shape[len(indices):]
            return self.slice(arg=indices+[(0, self.shape[i]) for i in range(len(indices), len(self.shape))]).reshape(shape=nshape)

        raise TypeError(
        f'list indices must be integers or slices, not {type(arg)}')


    @classmethod
    def zeros(cls, *shape, **kwargs):
        return cls(np.zeros(shape), **kwargs)

    @classmethod
    def ones(cls, *shape, **kwargs):
        return cls(np.ones(shape), **kwargs)

    @classmethod
    def diagonal(cls, dims, **kwargs):
        return cls(np.eye(dims), **kwargs)

    @classmethod
    def uniform(cls, *shape, low=-1.0, high=1.0, **kwargs):
        return cls(np.random.uniform(low, high, size=shape) 
                / np.sqrt(np.prod(shape)), **kwargs)

    @classmethod
    def normal(cls, *shape, loc=0.0, scale=1.0, **kwargs):
        return cls(np.random.normal(loc, scale, size=shape), **kwargs)

    @classmethod
    def full(cls, value, *shape, **kwargs):
        return cls(np.full(shape, value), **kwargs)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def chunk(self, chunks, dim=0):
        chunked = np.split(self.data, chunks, dim)

    def backward(self, allow_fill=True):
        if allow_fill:
            self.grad = None

        if self._ctx is None:
            return

        if self.grad is None and allow_fill:
            self.grad = np.ones(self.shape)

        parents = self._ctx.parents
        gradients = self._ctx.backward(self.grad, _idx=self._idx)
        if isinstance(gradients, np.ndarray):
            gradients = [gradients]

        for gradient, parent in zip(gradients, parents):
            if gradient is None:
                continue

            if parent.requires_grad:
                if parent.grad is None: 
                    gradient.flags.writeable = True
                    parent.grad = gradient
                else:
                    if not parent._isleaf:
                        parent.grad = gradient
                    else:
                        parent.grad += gradient
                parent.backward(allow_fill=False)

