# BPnetworks py3实现
BP神经网络的python实现，里面包含adam、RMSProp等多种优化算法.

网络结构等进行了参数化， 可通过参数设置实现网络结构等内容。

对于分类和回归，只需要修改反向传播过程中输出层的激活函数和dAL(L_model_backward)、前向传播过程中最后输出层的激活函数（L_model_forward）以及损失函数(cost_func)即可

另外对于超参数的选择，提供了hyperopt方法的使用。
