import numpy as np

class Dataset(object):
    def __init__(self, samples, labels):
        self._samples = samples
        self._labels = labels

    def __len__(self):
        return self._samples.shape[0]

    def __getitem__(self, idx):
        return self._samples[idx], self._labels[idx]

