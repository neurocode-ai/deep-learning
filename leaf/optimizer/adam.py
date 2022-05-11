import numpy as np
from leaf.optimizer import Optimizer

class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
            weight_decay=None):
        super(Adam, self).__init__(params)
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._weight_decay = weight_decay
        self._t = 0
        self._m = [np.zeros(t.shape) for t in self.params]
        self._v = [np.zeros(t.shape) for t in self.params]

    def step(self):
        self._t = self._t + 1.0
        alpha = self._lr * ((1.0 - self._beta2 ** self._t) ** 0.5) / (1.0 - self._beta1 **
                self._t)
        for i, p in enumerate(self.params):
            self._m[i] = self._beta1 * self._m[i] + (1.0 - self._beta1) * p.grad
            self._v[i] = self._beta2 * self._v[i] + (1.0 - self._beta2) * (p.grad ** 2)
            p.data -= (alpha * self._m[i]) / ((self._v[i] ** 0.5) + self._eps)

