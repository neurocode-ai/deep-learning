from leaf.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=None):
        super(SGD, self).__init__(params)
        self._lr = lr
        self._weight_decay = weight_decay

    def step(self):
        for p in self.params:
            p.data -= self._lr * p.grad

