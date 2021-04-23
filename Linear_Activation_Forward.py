import numpy as np
from code.Linear_Forward import *
from code.dnn_utils import *
from code.testCases import *


def linear_activation_forward(A_prev, W, b, activation):
    """
    实现LINEAR-> ACTIVATION 这一层的前向传播

    参数：
        A_prev - 来自上一层（或输入层）的激活，维度为(上一层的节点数量，示例数）
        W - 权重矩阵，numpy数组，维度为（当前层的节点数量，前一层的大小）
        b - 偏向量，numpy阵列，维度为（当前层的节点数量，1）
        activation - 选择在此层中使用的激活函数名，字符串类型，【"sigmoid" | "relu"】

    返回：
        A - 激活函数的输出，也称为激活后的值
        cache - 一个包含“linear_cache”和“activation_cache”的字典，我们需要存储它以有效地计算后向传递
    """

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    elif activation=='linef':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = linef(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


if __name__=='__main__':
    # 测试linear_activation_forward
    print("==============测试linear_activation_forward==============")
    A_prev, W, b = testCases.linear_activation_forward_test_case()

    A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="sigmoid")
    print("sigmoid，A = " + str(A))

    A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="relu")
    print("ReLU，A = " + str(A))

