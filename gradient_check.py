import numpy as np
from dnn_utils import *
import matplotlib.pyplot as plt
from init_para import *
from Update_Parameters import *
from L_model_forward import *
from L_model_backward import *
from cost_func import *
import time
from testCases import gradient_check_n_test_case
def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7):
    """
    检查backward_propagation_n是否正确计算forward_propagation_n输出的成本梯度
    参数：
        parameters - 包含参数“W1”，“b1”，“W2”，“b2”，“W3”，“b3”的python字典：
        grad_output_propagation_n的输出包含与参数相关的成本梯度。
        x  - 输入数据点，维度为（输入节点数量，1）
        y  - 标签
        epsilon  - 计算输入的微小偏移以计算近似梯度

    返回：
        difference - 近似梯度和后向传播梯度之间的差异
    """
    # 初始化参数
    parameters_values= dictionary_to_vector(parameters)   #47x1
    grad = gradients_to_vector(gradients)#47x1
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    # 计算gradapprox
    for i in range(num_parameters):
        # 计算J_plus [i]。输入：“parameters_values，epsilon”。输出=“J_plus [i]”
        thetaplus = np.copy(parameters_values)  # Step 1
        thetaplus[i][0] = thetaplus[i][0] + epsilon  # Step 2
        AL, caches = L_model_forward(X, vector_to_dictionary(thetaplus))
        J_plus[i]=compute_cost(AL, Y, vector_to_dictionary(thetaplus), 4, lambd=0)

        # 计算J_minus [i]。输入：“parameters_values，epsilon”。输出=“J_minus [i]”。
        thetaminus = np.copy(parameters_values)  # Step 1
        thetaminus[i][0] = thetaminus[i][0] - epsilon  # Step 2
        AL, caches = L_model_forward(X, vector_to_dictionary(thetaminus))
        J_minus[i] = compute_cost(AL, Y, vector_to_dictionary(thetaminus), 4, lambd=0)

        # 计算gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon) #47x1


    # 通过计算差异比较gradapprox和后向传播梯度。
    numerator = np.linalg.norm(grad - gradapprox)  # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)  # Step 2'
    difference = numerator / denominator  # Step 3'

    if difference < 1e-7:
        print("梯度检查：梯度正常!")
    else:
        print("梯度检查：梯度超出阈值!")

    return difference
if __name__=='__main__':
    X,Y,parameters=gradient_check_n_test_case()
    AL, caches = L_model_forward(X, parameters)
    grads = L_model_backward(AL, Y, caches, lambd=0)
    grad={}
    for l in range(1,4):
        grad['dW1'] = grads['dW1']
        grad['dW2'] = grads['dW2']
        grad['dW3'] = grads['dW3']
        grad['db1'] = grads['db1']
        grad['db2'] = grads['db2']
        grad['db3'] = grads['db3']

    parameters_value=dictionary_to_vector(parameters)
    difference= gradient_check_n(parameters, grad, X, Y, epsilon=1e-7)
    print(difference)


