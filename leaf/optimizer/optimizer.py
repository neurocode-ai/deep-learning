from leaf import Tensor

class Optimizer(object):
    def __init__(self, params):
        assert all(isinstance(p, Tensor) for p in params)
        self.params = [p for p in params if p.requires_grad]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError(
        f'user defined Optimizer has not implemented gradient update')


