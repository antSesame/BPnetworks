# BPnetworks
BP神经网络的python实现，里面包含adam、RMSProp等多种优化算法.

网络结构参数化实现。

对于分类和回归，只需要修改反向传播过程中输出层的激活函数和dAL(L_model_backward)、前向传播过程中最后输出层的激活函数（L_model_forward）以及损失函数(cost_func)即可
