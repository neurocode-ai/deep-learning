# Deep learning, with education in focus
With a passion rooted in knowledge, learning, and sharing of information, this repository aims to provide extensive information on majority of things deep learning related. This repository is for anyone interested in deep learning, with extensive guides and tutorial notebooks designed for both the curious novice learner and the experienced practitioner. As such, this is a Python focused repository that provides three high-level features:
- Lecture notebooks on artificial neural networks and deep learning
- Tutorials to evaluate your theoretical understanding on practical tasks
- Minimalistic deep learning framework (with NumPy)

## Lectures
tldr; neural network notebook lectures

## Tutorials
tldr; deep learning fundamentals in practice

## leaf, the minimal deep learning framework
In its essence, it is a Python only implementation of the PyTorch autograd framework. However, leaf is intended to be streamlined whilst offering a minimal and robust api, everything built using NumPy. Since education is at the core for why leaf was developed, the most fundamental part has, and will always be, intuitive and easy-to-read documentation for all building blocks of the deep learning framwork. Nonetheless, competitive performance will be difficult to obtain with leaf, and as such, leaf is only intended to be used for educational purposes. 

### Example
```python
from leaf import Tensor

x = Tensor([[-1.4, 2.5, 7.8]], requires_grad=True)
w = Tensor.eye(3, requires_grad=True)

y = x.dot(w).mean()
y.backward()

print(x.grad)  #dy/dx
print(w.grad)  #dy/dw
```

### Neural networks
The autograd library is basically all you need to build a barebones deep learning framework. Implement some criteria, optimizers, and boilerplate mini-batching code on top of that and you have everything you need to build neural networks! With leaf you can implement neural networks like you would in PyTorch, but compile them and train without all the boilerplate code.
```python
import numpy as np
import leaf.nn as nn

# define your neural network, PyTorch style
class MNISTNet(nn.Module):
  def __init___(self):
    self.layers = nn.Sequential(
      nn.Linear(784, 128),
      nn.ReLU(),
      nn.Linear(128, 10),
      nn.LogSoftmax()
    )
  
  # implement forward pass for invoking __call__
  def forward(self, x):
    return self.layers(x)
```
Next just initialize the criteria for you domain-specific task, dataloader, optimizer of your choice, and you can seamlessly train a neural network from scratch. It supports training models with boilerplate code as well, like in PyTorch, but preferred and easier way is to use the trainer class.
```python
model = MNISTNet()
criterion = leaf.criterion.NLLLoss()
optimizer = leaf.optim.Adam(
  model.parameters(), lr=1e-3, weight_decay=1e-5
)

dataloader = leaf.datautil.fetch_mnist()
trainer = leaf.Trainer(
  model,
  criterion,
  optimizer,
  dataloader,
  max_epochs=100
)
```

### Installation
Currently you will have to clone this repository and move the files to your working directory. Implementing `setup.py` is work in progress.

### Running tests
Use the bash script. You might have to provide executable rights to the bash script ...
```bash
$ sudo chmod +x test.sh
$ ./test.sh
```
... or just start the unittest module with Python directly, in the root directory.
```
$ python3 -m unittest
```

## License
Neurocode repositories has an Apache-2.0 styled license, please see [LICENSE](https://github.com/NEUROCODE-ai/edugrad/blob/master/LICENSE) for more information.

