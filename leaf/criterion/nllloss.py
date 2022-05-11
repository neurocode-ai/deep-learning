import numpy as np
from leaf import Tensor
from .criterion import Criterion

class NLLLoss(Criterion):
    def apply(self, logits, targets):
        n_classes = logits.shape[-1]
        y = np.zeros((targets.shape[0], n_classes)).astype(np.float32)
        y[range(y.shape[0]), targets.data.astype(int)] = -1.0 * n_classes
        return logits.multiply(Tensor(y)).mean()

