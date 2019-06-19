"""

@file   : 10-手动实现反向传播算法.py

@author : xiaolu

@time   : 2019-06-19

"""
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1)
x = np.linspace(-1, 1, 200)[:, None]  # 相当于扩展了一维  此时理解为[batch, 1]
y = x**2

# 设置学习率
learning_rate = 0.001


def tanh(x):
    # 实现激活函数
    return np.tanh(x)


def derivative_tanh(x):
    # 激活函数的导数
    return 1 - tanh(x) ** 2


w1 = np.random.uniform(0, 1, (1, 10))
w2 = np.random.uniform(0, 1, (10, 10))
w3 = np.random.uniform(0, 1, (10, 1))

b1 = np.full((1, 10), 0.1)   # 生成array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
b2 = np.full((1, 10), 0.1)
b3 = np.full((1, 1), 0.1)


for i in range(300):
    a1 = x
    z2 = a1.dot(w1) + b1   # (200, 10) 相当于全连接   横向往下传播+b
    a2 = tanh(z2)    # 激活层 (200, 10)
    z3 = a2.dot(w2) + b2   # (200, 10) 再接全连接层   横向往下传播+b
    a3 = tanh(z3)    # (200, 10)
    z4 = a3.dot(w3) + b3    # (200, 1)

    cost = np.sum((z4 - y)**2) / 2

    # 单向传播
    z4_delta = z4 - y
    dw3 = a3.T.dot(z4_delta)
    db3 = np.sum(z4_delta, axis=0, keepdims=True)

    z3_delta = z4_delta.dot(w3.T) * derivative_tanh(z3)
    dw2 = a2.T.dot(z3_delta)
    db2 = np.sum(z3_delta, axis=0, keepdims=True)   # keepdims保留原始的维度 不然的话numpy会根据你的数据自动化设置维度

    z2_delta = z3_delta.dot(w2.T) * derivative_tanh(z2)
    dw1 = x.T.dot(z2_delta)
    db1 = np.sum(z2_delta, axis=0, keepdims=True)
    for param, gradient in zip([w1, w2, w3, b1, b2, b3], [dw1, dw2, dw3, db1, db2, db3]):
        param -= learning_rate * gradient

    print(cost)

plt.plot(x, y, c='red')
plt.plot(x, z4, c='blue')
plt.show()