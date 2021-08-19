import numpy as np

def Conv(data, n=10):
    x = data[1:].astype(np.float32) / 255
    y = np.array([int(i == data[0]) for i in range(n)]).astype(np.float32)
    return (x, y)

def load_data(path):
    data = np.loadtxt(path, delimiter=',', dtype=np.int)
    ret = [*data]
    for i in range(len(ret)): ret[i] = Conv(ret[i])
    return ret
