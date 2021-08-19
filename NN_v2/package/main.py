from modules import *

if __name__ == '__main__':
    train_data = mnist_loader.load_data('../../data/train_data_50000.csv')
    test_data  = mnist_loader.load_data('../../data/test_data_10000.csv')
    N = NN_v2.Network([784, 28, 28, 10])
    N.SGD(30, 10, 0.03, 0.1, train_data, test_data=test_data)
