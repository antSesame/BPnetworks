import numpy as np


def initialize_parameters_deep(layers_dims):
    """
    此函数是为了初始化多层网络参数而使用的函数。
    参数：
        layers_dims - 包含我们网络中每个图层的节点数量的列表

    返回：
        parameters - 包含参数“W1”，“b1”，...，“WL”，“bL”的字典：
                     W1 - 权重矩阵，维度为（layers_dims [1]，layers_dims [1-1]）
                     bl - 偏向量，维度为（layers_dims [1]，1）
    """
    np.random.seed()
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1])*np.sqrt(2/layers_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

    # 确保我要的数据的格式是正确的
        assert (parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
        assert (parameters["b" + str(l)].shape == (layers_dims[l], 1))


    return parameters

if __name__=='__main__':
    # 测试initialize_parameters_deep
    print("==============测试initialize_parameters_deep==============")
    layers_dims = [5, 4, 3]
    parameters = initialize_parameters_deep(layers_dims)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    a=np.random.rand(10)
    b=10**a
    print(a,b)

    print(np.arange(1,11))






