# NN_v2

- Multi-Layer Feed-Forward Neural Network with back propagation

- Improved performance from NN_v1

- used ReLU activation function

- used CrossEntropy cost function

- He initialization applied

- L2 Regularization applied

- increased train data : (9000, 1000) -> (50000, 10000)

## Example code

```
from modules import *

if __name__ == '__main__':
    train_data = mnist_loader.load_data('../data/train_data_50000.csv')
    test_data  = mnist_loader.load_data('../data/test_data_10000.csv')
    N = NN_v2.Network([784, 28, 28, 10])
    N.SGD(30, 10, 0.03, 0.1, train_data, test_data=test_data)
```

## Accuracy

- 96.6 ~ 96.9 %

- 97.4 ~ 97.5 % with Data Augmentation [[code]](https://github.com/jinhan814/NN_implementation/blob/main/NN_v2/NN_v2.ipynb)

## c++ code

- Handwritten Digits Classification implemented by c++

- used same MLP model as NN_v2, w and b are initialized by trained w, b of main.py

- This code is solution code for [BOJ 18824](https://www.acmicpc.net/problem/18824).

- Blog post : [https://blog.naver.com/jinhan814/222475395350](https://blog.naver.com/jinhan814/222475395350)

#### input example

```
1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 8 67 127 160 234 191 143 13 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 27 150 229 254 254 254 254 254 254 182 12 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 2 59 237 254 254 247 185 150 150 150 237 234 23 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 138 254 254 254 178 22 0 0 0 0 15 190 225 34 0 0 0 0 0
0 0 0 0 0 0 0 23 169 252 254 238 60 3 0 0 0 0 15 121 254 254 190 0 0 0 0 0
0 0 0 0 0 0 0 111 254 254 231 31 0 0 0 0 0 47 190 254 254 247 78 0 0 0 0 0
0 0 0 0 0 0 0 198 254 238 59 0 0 0 0 0 94 209 254 254 254 103 0 0 0 0 0 0
0 0 0 0 0 0 9 210 254 206 0 0 0 49 127 211 249 254 254 254 235 3 0 0 0 0 0 0
0 0 0 0 0 0 6 206 254 247 217 217 217 248 254 254 254 254 254 254 143 0 0 0 0 0 0 0
0 0 0 0 0 0 0 128 254 254 254 254 254 254 254 255 254 254 254 218 12 0 0 0 0 0 0 0
0 0 0 0 0 0 0 4 123 188 238 188 167 95 95 67 244 254 235 49 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 21 0 0 0 0 174 254 254 164 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 113 254 254 208 19 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 9 208 254 254 87 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 133 254 254 166 4 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 54 247 254 250 43 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 217 254 254 157 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 104 253 254 254 4 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 17 243 254 254 162 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 5 163 254 128 17 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```

#### output example

```
9
```
