""" ---------------------------------------------------------------------------
Definition and implementation of the Tensor class, the fundamental
building block of the autograd framework. Initializing the class
is either done by prividing an ndarray (NumPy) or specifying what
pre-initialized data array that should be used.

If any Tensor is used created, it is referred to as a leaf node. This means
that it should be a root node in the directed acyclic graph which is created
when invoking a backward pass. Leaf nodes are by default set to NOT require
gradient calculation. All parameters of a nn.Module has requires_gradient=True,
but can be set to False by user.

All Tensors that are not a leaf node should have been created from a
Function, as such, they are part of a context. A context is the result
of an invoked Function call, e.g. `add` or `matmul`, and stores the parent
nodes of the created Tensor. Furthermore, the context is used to propagate
the gradient during the backwards pass, and each Function (context) has to
have a backwards pass implementation for this to be possible. For more 
information on Functions and contexts, see `leaf/functions/*` directory.


Authors: Wilhelm Ågren <wilhelmagren98@gmail.com>
Last edited: 18-05-2022
License: Aapache 2.0
--------------------------------------------------------------------------- """
import numpy as np

def concatenate(tensors, dim=0):
    """ temporary implementation of the concatnate Function... """
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

    def __add__(self, t):
        return self.add(t)

    def __sub__(self, t):
        return self.sub(t)
    
    def __mul__(self, t):
        return self.mul(t)

    def __div__(self, t):
        return self.div(t)

    def __repr__(self):
        return f'<leaf.Tensor(\n{self.data}\n' \
                f'dtype={self.dtype}, grad_fn={self._ctx}, grad={self.grad}>'

    def __getitem__(self, indices):
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
    
    def detach(self):
        return Tensor(self.data, dtype=self.dtype, requires_grad=False)

    def _topological_sort(self):
        def _recursive_walk(node, visited, nodes):
            visited.add(node)
            if node._ctx:
                [_recursive_walk(n, visited, nodes) 
                        for n in node._ctx.parents if n not in visited]
                nodes.append(node)
            return nodes
        return _recursive_walk(self, set(), [])

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

        for t in reversed(self._topological_sort()):
            if not any(tt.requires_grad for tt in t._ctx.parents):
                continue

            assert t.grad is not None, \
            'The parents of the current node requires gradient, but no gradient has ' \
            'been calculated or initialized for the current node. Make sure that the ' \
            'current node inherited the correct grad requiremnts from parents.'
            
            parents = t._ctx.parents
            gradients = t._ctx.backward(t.grad, _idx=t._idx)

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

