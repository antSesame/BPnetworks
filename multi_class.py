from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

from code.L_model_forward import L_model_forward
from code.dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward #参见资料包
import code.lr_utils #参见资料包，或者在文章底部copy
from code.L_layer_model import *
from code.init_para import initialize_parameters_deep

data = loadmat('ex4data1.mat')
X=data['X']
Y=data['y']
train_x=X.T # 400x500
yk = np.zeros((5000, 10))
for num in range(Y.size):
    yk[num, Y[num] - 1] = 1
train_y=yk.T # 10x5000
layers_dims = [400, 25, 10] #  5-layer model
parameters = initialize_parameters_deep(layers_dims)

parameters = L_layer_model(train_x, train_y, layers_dims, lambd=1,learning_rate=0.1,mini_batch_size=2048,num_iterations = 1000,optimizer='adam', print_cost = True,isPlot=True)

pred=L_model_forward(train_x, parameters)
pred=pred[0].T
print(pred.shape)
P = np.zeros(5000)
for num in range(5000):
		# 找到第num行中，与该行最大值相等的列的下标，此时下标的范围是[0,9]
		# label的范围是[1,10]，需要把下标的值+1
		# np.where()返回的是一个长度为2的元祖，保存的是满足条件的下标
		# 元组中第一个元素保存的是行下标，第二元素保存的是列下标
	index = np.where(pred[num,:] == np.max(pred[num,:]))
	P[num] = index[0][0].astype(int) + 1

Y=Y.ravel()
print(np.mean(P == Y)*100)