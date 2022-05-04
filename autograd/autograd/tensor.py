import numpy as np

class Tensor(object):
    """ Wrapping numpy.ndarray for forward/backward pass operations.
    Initialize object with either of (int, float, list, tuple, np.ndarray),
    specify if the Tensor is going to require a gradient leaf to be memorized
    during forward pass for the backward pass, determine dtype of Tensor.

    Example usage
    -------------
    >>> tensor1 = Tensor(np.ones((5,1,4)), dtype=np.float64)
    >>> tensor2 = Tensor([[-4.0, -1, 15.2]], requires_grad=True)

    Init parameters
    ---------------
    data: int | float | list | tuple | np.ndarray
        The data modality to wrap as a Tensor, if the provided data is not
        valid, the __init__ func raises a ValueError and exits.
    requires_grad: bool
        True/False whether or not to require gradient in backwards pass.
    dtype: int | float | np.dtype
        The data type object to cast the input data as. It will subsequentially
        be the dtype of the initialized Tensor object.

    """
    def __init__(self, data, requires_grad=False, dtype=np.float32):
        self.requires_grad = requires_grad

        if isinstance(data, int) or isinstance(data, float):
            data = np.array([data]).astype(dtype)

        elif isinstance(data, list) or isinstance(data, tuple):
            data = np.array(data).astype(dtype)

        elif isinstance(data, np.ndarray):
            data = data.astype(dtype)

        else: raise ValueError(
                f'unknown data instance passed to Tensor, {type(data)}')

        self.data = data
        self.grad = None
        self._ctx = None

    def __str__(self):
        return f'<autograd.Tensor\n{self.data}\n' \
                f'dtype={self.dtype}, grad_fn={self._ctx}, grad={self.grad}>'

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype
        
    def backward(self, allow_fill=True):
        """ Recursively perform gradient backwards pass for
        all parents with requires_grad=True from this current Tensor. 
        Call ends when either at user created Tensors, i.e., no 
        function context tied to the Tensor, or, the Tensor does not
        require grad.

        Example usage
        -------------
        >>> x = Tensor(np.ones((128, 784)))
        >>> W = Tensor(np.ones((784, 64)), requires_grad=True)
        >>> y = x.dot(W).mean()
        >>> y.backward()

        Parameters
        ----------
        allow_fill: bool
            Should be False for all Tensors that are no the one
            invoking the initial backwards call. For the first
            gradient, all values are set to 1, since
            grad_X/grad_X == 1, so fill the gradient with ones in
            shape of current Tensor.

        """
        if not self._ctx:
            return
        
        if self.grad is None and allow_fill:
            assert self.shape == (1, )
            self.grad = np.ones_like(self.data).reshape((-1, 1))

        parents = self._ctx.parents
        grads = self._ctx.backward(self.grad)
        grads = [grads] if len(parents) == 1 else grads

        for grad, parent in zip(grads, parents):
            if grad is None:
                continue

            if parent.requires_grad:
                parent.grad = grad

            parent.backward(allow_fill=False)

