# FFNN_v1

✏ Python implementation of multi-layer feed forward neural networks with back propagation.

- Used NumPy package for matrix operations.

- No optimization is applied.

- All code is based on ["Neural Networks and Deep Learning"](https://github.com/mnielsen/neural-networks-and-deep-learning) by Michael Nielsen.

## Example code

```
from modules import *

if __name__ == '__main__':
    train_data, test_data = mnist_loader.load_data()
    N = FFNN_v1.Network([784, 28, 28, 10])
    N.SGD(30, 10, 3.0, train_data, test_data = test_data)
```

## Accuracy

- 90.1 ~ 91.4%