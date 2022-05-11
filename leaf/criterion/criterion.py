class Criterion(object):
    def __call__(self, preds, targets):
        return self.apply(preds, targets)

    def apply(self, *args, **kwargs):
        raise NotImplementedError(
        f'user defined Criterion has not implemented apply method')

