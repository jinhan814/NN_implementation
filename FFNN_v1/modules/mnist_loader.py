import numpy as np

def Conv(data, n=10):
    x = data[1:].astype(np.float32) / 255
    y = np.array([int(i == data[0]) for i in range(n)]).astype(np.float32)
    return (x, y)

def load_data():
    data = np.loadtxt('../data/mnist_test.csv', delimiter=',', dtype=np.int)
    train_data, test_data = [*data[:9000]], [*data[9000:]]
    for i in range(len(train_data)): train_data[i] = Conv(train_data[i])
    for i in range(len(test_data)): test_data[i] = Conv(test_data[i])
    return (train_data, test_data)
