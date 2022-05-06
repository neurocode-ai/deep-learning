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

### Installation
Currently you will have to clone this repository and move the files to your working directory. Implementing `setup.py` is work in progress.

### Running tests
Either run ```$ python3 -m unittest``` when in the root directory, or run the bash script ```test.sh``` after granting executable rights.

## License
EDUGRAD has an Apache-2.0 styled license, please see [LICENSE](https://github.com/NEUROCODE-ai/edugrad/blob/master/LICENSE) for more information.

