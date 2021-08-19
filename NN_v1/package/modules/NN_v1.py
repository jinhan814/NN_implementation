import numpy as np
import random


class Sigmoid:
    def __call__(self, x):
        return 1. / (1. + np.exp(-x))

    def deriv(self, x):
        return self(x) * (1 - self(x))


class MSE:
    def __call__(self, res, y):
        return sum((i - j) * (i - j) for i, j in zip(res, y))

    def deriv(self, res, y):
        return res - y


class Network:
    def __init__(self, shape, activation_f=Sigmoid(), cost_f=MSE()):
        np.random.seed(2048)
        self.shape = shape
        self.w = [np.random.randn(y, x) for x, y in zip(shape[:-1], shape[1:])]
        self.b = [np.random.randn(y, 1) for y in shape[1:]]
        self.activation_f = activation_f
        self.cost_f = cost_f

    def forward(self, x):
        ret = x.reshape(-1, 1)
        for w, b in zip(self.w, self.b):
            ret = self.activation_f(np.dot(w, ret) + b)
        return ret.reshape(-1)

    def backward(self, _x, _y):
        dw = [np.zeros(w.shape) for w in self.w]
        db = [np.zeros(b.shape) for b in self.b]
        x, y = _x.reshape(-1, 1), _y.reshape(-1, 1)
        a, z = [x], []

        for w, b in zip(self.w, self.b):
            x = np.dot(w, x) + b
            z.append(x)
            x = self.activation_f(x)
            a.append(x)

        dz = self.cost_f.deriv(a[-1], y) * self.activation_f.deriv(z[-1])
        dw[-1] = np.dot(dz, a[-2].transpose())
        db[-1] = dz

        for i in range(2, len(self.shape)):
            dz = np.dot(self.w[-(i - 1)].transpose(), dz) * self.activation_f.deriv(z[-i])
            dw[-i] = np.dot(dz, a[-(i + 1)].transpose())
            db[-i] = dz

        return (dw, db)

    def update(self, batch, lr):
        dw = [np.zeros(w.shape) for w in self.w]
        db = [np.zeros(b.shape) for b in self.b]

        for x, y in batch:
            _dw, _db = self.backward(x, y)
            dw = [w + _w for w, _w in zip(dw, _dw)]
            db = [b + _b for b, _b in zip(db, _db)]

        self.w = [w - (lr / len(batch)) * _w for w, _w in zip(self.w, dw)]
        self.b = [b - (lr / len(batch)) * _b for b, _b in zip(self.b, db)]

    def SGD(self, epochs, batch_size, lr, train_data, test_data=None):
        for epoch in range(1, epochs + 1):
            random.Random(epoch).shuffle(train_data)
            batchs = [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)]
            for batch in batchs: self.update(batch, lr)
            if test_data: print(f"Epoch : {epoch}, Evaluate : {self.evaluate(test_data)} / {len(test_data)}")
            else: print(f"Epoch : {epoch}")

    def evaluate(self, test_data):
        ret = sum(int(np.argmax(self.forward(x)) == np.argmax(y)) for x, y in test_data)
        return ret
