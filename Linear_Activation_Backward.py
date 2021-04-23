from code.Linear_Backward import *
from code.dnn_utils import *


def linear_activation_backward(dA, cache,lambd,activation="relu"):
    """
    实现LINEAR-> ACTIVATION层的后向传播。

    参数：
         dA - 当前层l的激活后的梯度值
         cache - 我们存储的用于有效计算反向传播的值的元组（值为linear_cache，activation_cache）
         activation - 要在此层中使用的激活函数名，字符串类型，【"sigmoid" | "relu"】
    返回：
         dA_prev - 相对于激活（前一层l-1）的成本梯度值，与A_prev维度相同
         dW - 相对于W（当前层l）的成本梯度值，与W的维度相同
         db - 相对于b（当前层l）的成本梯度值，与b的维度相同
    """
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache,lambd)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache,lambd)
    elif activation=='linef':
        dZ = linef_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache,lambd)

    return dA_prev, dW, db


if __name__=='__main__':
    AL, linear_activation_cache = linear_activation_backward_test_case()

    dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation="sigmoid",lambd =0)
    print("sigmoid:")
    print("dA_prev = " + str(dA_prev))
    print("dW = " + str(dW))
    print("db = " + str(db) + "\n")

    dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation="relu",lambd =0)
    print("relu:")
    print("dA_prev = " + str(dA_prev))
    print("dW = " + str(dW))
    print("db = " ,db)