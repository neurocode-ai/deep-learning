import numpy as np
from leaf import Tensor

def _default_collate_fn(*items):
    return tuple(Tensor(item) for item in items)

class Dataset(object):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, collate_fn=None, seed=1):
        assert batch_size > 0
        assert isinstance(batch_size, int)
        assert isinstance(dataset, Dataset)

        if collate_fn is None:
            collate_fn = _default_collate_fn

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.iterlen = len(dataset) // batch_size
        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)

    def __len__(self):
        return self.iterlen
    
    def __iter__(self):
        for i in range(len(self)):
            if self.shuffle:
                indices = self.rng.choice(len(self.dataset), self.batch_size, replace=False)
                yield self.collate_fn(*self.dataset[indices])
                continue
            
            lowidx, upidx = i * self.batch_size, (i + 1) * self.batch_size
            yield self.collate_fn(self.dataset[lowidx:upidx])

