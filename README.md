# FFNN_implementation

✏ Feed-Forward Neural Network Implementation using Numpy

Python implementation of multi-layer feed forward neural networks with back propagation. No optimization is applied.

The code is based on ["Neural Networks and Deep Learning"](https://github.com/mnielsen/neural-networks-and-deep-learning) by Michael Nielsen.

## Example code

```
N = Network([784, 28, 28, 10])

N.SGD(30, 10, 3.0, train_data, test_data = test_data)
```

## Update Notes

* 21-08-16 : add implementation using numpy