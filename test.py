import codecs
import json

import pandas as pd
import numpy as np
from code.L_layer_model import *
import matplotlib.pyplot as plt
from code.predict import *

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def normal(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_normal = (X - X_mean) / (X_std + 1e-8)
    return X_normal, X_mean, X_std


data = pd.read_excel('../data/hadle_no_miss.xlsx')  # (1424, 27)
x = np.array(data.iloc[:, 6:21])  # (1424, 15)

y = np.array(data.iloc[:, 21:])
assert y.shape == (1424, 6)

train_x = x[:1000, :].T
train_y = y[:1000, :].T
assert train_x.shape == (15, 1000)
X_normal, X_mean, X_std = normal(train_x.T)
Y_normal, Y_mean, Y_std = normal(train_y.T)

train_x = X_normal.T
train_y = Y_normal.T

test_x = x[1000:, :]
test_x = (test_x - X_mean) / (X_std + 1e-8)  # 标准化
test_x = test_x.T
test_y = y[1000:, :]
test_y = (test_y - Y_mean) / (Y_std + 1e-8)
test_y = test_y.T
assert test_x.shape == (15, 424)

layers_dims = [15, 10, 10, 10, 6]

parameters3 = L_layer_model(train_x, train_y, layers_dims, learning_rate=0.002567161349434759, mini_batch_size=16,
                            lambd=0.01, num_iterations=5000,
                            beta=0.9, optimizer='adam', print_cost=True, isPlot=True, over_stop=False)




class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


# 字典写入文件
with codecs.open("../data/para_best.csv", 'w', encoding='utf-8') as f:
    json.dump(parameters3[0], f, cls=NpEncoder)


# 读取字典
file = open("../data/para_best.csv", 'r')

js = file.read()
dict_num = json.loads(js)
for i in dict_num.keys():
    dict_num[i] = np.array(dict_num[i])





cost1, error, testy2 = predict(test_x, test_y, parameters3[0], L=len(layers_dims), lambd=0.01)

cost2, error, testy2 = predict(test_x, test_y, dict_num, L=len(layers_dims), lambd=0.01)
print(cost1 , cost2)
end = testy2.T * (Y_std + 1e-8) + Y_mean
print(end.shape)
error = np.mean(abs(end - y[1000:, :]), axis=0)

print(error)




plt.show()



