import numpy as np
from code.L_model_forward import *
from code.cost_func import *


def predict(X, y, parameters, L, lambd=0):
    """
    该函数用于预测L层神经网络的结果，当然也包含两层

    参数：
     X - 测试集
     y - 标签
     parameters - 训练模型的参数

    返回：
     p - 给定数据集X的预测
    """

    m = X.shape[1]
    n = len(parameters) // 2  # 神经网络的层数
    #

    # 根据参数前向传播
    probas, caches = L_model_forward(X, parameters)
    # probas=np.sum(probas-y)
    cost = compute_cost(probas, y, parameters, L, lambd)
    error = np.mean(abs(probas - y), axis=1)
    # error=np.sum(abs(probas-y),axis=1)/y.shape[1]
    return cost, error, probas


def getError(AL, y):
    error = np.sum(abs(AL - y)) / y.shape[1]
    error = np.squeeze(error)
    return error

    # 用于分类预测
    # for i in range(0, probas.shape[1]):
    #     if probas[0, i] > 0.5:
    #         p[0, i] = 1
    #     else:
    #         p[0, i] = 0
    #
    # print("准确度为: " + str(float(np.sum((p == y)) / m)))

    # return p
