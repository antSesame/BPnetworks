import numpy as np
def initialize_RMSprop(parameters):
    """
    初始化速度，velocity是一个字典：
        - keys: "dW1", "db1", ..., "dWL", "dbL" 
        - values:与相应的梯度/参数维度相同的值为零的矩阵。
    参数：
        parameters - 一个字典，包含了以下参数：
            parameters["W" + str(l)] = Wl
            parameters["b" + str(l)] = bl
    返回:
        v - 一个字典变量，包含了以下参数：
            v["dW" + str(l)] = dWl的速度
            v["db" + str(l)] = dbl的速度

    """
    L = len(parameters) // 2  # 神经网络的层数
    s = {}

    for l in range(L):
        s["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        s["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

    return s


def update_parameters_with_RMSprop(parameters, grads, s, beta, learning_rate,epslion=1e-8):
    """
    使用动量更新参数
    参数：
        parameters - 一个字典类型的变量，包含了以下字段：
            parameters["W" + str(l)] = Wl
            parameters["b" + str(l)] = bl
        grads - 一个包含梯度值的字典变量，具有以下字段：
            grads["dW" + str(l)] = dWl
            grads["db" + str(l)] = dbl
        v - 包含当前速度的字典变量，具有以下字段：
            v["dW" + str(l)] = ...
            v["db" + str(l)] = ...
        beta - 超参数，动量，实数
        learning_rate - 学习率，实数
    返回：
        parameters - 更新后的参数字典
        v - 包含了更新后的速度变量
    """
    L = len(parameters) // 2
    for l in range(L):
        # 计算速度
        s["dW" + str(l + 1)] = beta * s["dW" + str(l + 1)] + np.square(grads["dW" + str(l + 1)])
        s["db" + str(l + 1)] = beta * s["db" + str(l + 1)] +np.square(grads["db" + str(l + 1)])

        # 更新参数
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]/(np.sqrt(s["dW" + str(l + 1)])+epslion)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]/(np.sqrt(s["db" + str(l + 1)])+epslion)

    return parameters, s
