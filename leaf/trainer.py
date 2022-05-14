import leaf.nn as nn
from leaf.optimizer import Optimizer
from leaf.criterion import Criterion
from leaf.datautil import DataLoader

import matplotlib.pyplot as plt

class Score(object):
    def __init__(self, tpe):
        self.tpe = tpe
        self.scores = {'train': [], 'valid': []}

    def push(self, key, *scores):
        self.scores[key].extend(scores)

    def plot(self):
        plt.plot(self.scores['train'], label=f'training {self.tpe}')
        plt.plot(self.scores['valid'], label=f'validation {self.tpe}')
        plt.legend()
        plt.show()

class Trainer(object):
    def __init__(self, trainloader, testloader=None, max_epochs=100, **kwargs):
        assert max_epochs > 0
        assert isinstance(trainloader, DataLoader)
        if testloader is not None:
            assert isinstance(testloader, DataLoader)

        print(trainloader)
        print(testloader)
        self.trainloader = trainloader
        self.testloader = testloader
        self.max_epochs = max_epochs
        
    def fit(self, model, optimizer, criterion):
        assert isinstance(model, (nn.Module, nn.Sequential))
        assert isinstance(optimizer, Optimizer)
        assert isinstance(criterion, Criterion)

        hist_loss = Score('loss')
        hist_acc = Score('acc')

        for epoch in range(self.max_epochs):
            for (samples, labels) in self.trainloader:
                logits = model(samples)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                hist_loss.push('train', loss.data[0][0])

            if self.testloader is not None:
                for (samples, labels) in self.testloader:
                    logits = model(samples)
                    loss = criterion(logits, labels)

                    hist_loss.push('valid', loss.data[0][0])

        return hist_loss, hist_acc

