from leaf import Tensor
from .criterion import Criterion

class MSELoss(Criterion):
    def apply(self, preds, targets):
        diff = preds.sub(targets)
        return diff.pow(Tensor(2)).mean()

