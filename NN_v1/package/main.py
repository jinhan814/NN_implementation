from modules import *

if __name__ == '__main__':
    train_data, test_data = mnist_loader.load_data('../../data/mnist_test.csv')
    N = NN_v1.Network([784, 28, 28, 10])
    N.SGD(50, 10, 1.0, train_data, test_data = test_data)
