import numpy as np

def concatenate(tensors, dim=0):
    tmp = Tensor.zeros(1)
    return tmp.concatenate(*tensors, dim=dim)

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

    def __repr__(self):
        return f'<leaf.Tensor(\n{self.data}\n' \
                f'dtype={self.dtype}, grad_fn={self._ctx}, grad={self.grad}>'

    def __getitem__(self, indices):
        """ Access items in Tensor.data via array indexing or slicing.
        
        if you invoke tensor[0], indices is type int
        if you invoke tensor[:2], indices is type slice
            a slice has attributes start, stop, step, in this case would be
            start=0, stop=2, step=1
        if you invoke tensor[1:4:2], start=1, stop=4, step=2, which would yield 
            output of length 2, since you get tensor[1] and tensor[3], the stop
            attribute is not included, so in range [slice.start, slice.stop)

        if you invoke tensor[:, 0, :], indices is type tuple
        because you now have three getitems that you want to do basically
        the tuple is := (slice, int, slice)
        with slices having start=0, stop=len(dim), step=1
        
        """
        
        args = []
        shape = []

        if indices is not None:
            for idx, s in enumerate(indices if isinstance(indices, (list, tuple)) else [indices]):
                if isinstance(s, (int, float)):
                    args.append((s, s + 1))
                    continue

                start_idx = s.start if s.start is not None else 0
                stop_idx = (s.stop if s.stop >= 0 else self.shape[idx] + s.stop) if s.stop is not None else self.shape[idx]
                args.append((start_idx, stop_idx))  # the range in the dim that we want to access
                shape.append(args[-1][1] - args[-1][0])  # basically stop - start

            # add any dims that are not being accessed
            shape += self.shape[len(args):]
            return self.slice(arg=args+[(0, self.shape[i]) for i in range(len(args),
                len(self.shape))], nshape=shape)

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

    def backward(self, allow_fill=True):
        if allow_fill:
            self.grad = None

        if self._ctx is None:
            return

        if self.grad is None and allow_fill:
            assert np.prod(self.shape) == 1, \
            'You are trying to initiate a backwards call on a Tensor that ' \
            'has not been reduced yet. The expected behavior is to reduce ' \
            'your Tensor using either .sum() or .mean(), otherwise the implicit ' \
            'creation of the gradient, i.e. initializing it with ones, might be ' \
            'incorrect.'

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

