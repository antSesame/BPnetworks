import codecs
import json


import pandas as pd
import numpy as np
from code.L_layer_model import *
import matplotlib.pyplot as plt
from code.predict import *
from code.csvRW import *

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def normal(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_normal = (X - X_mean) / (X_std + 1e-8)
    return X_normal, X_mean, X_std


data = pd.read_excel('../data/hadle_no_miss.xlsx')  # (1424, 27)
x = np.array(data.iloc[:, 6:21])  # (1424, 15)

y = np.array(data.iloc[:, 21:22])
assert y.shape == (1424, 1)

train_x = x[:1000, :].T
train_y = y[:1000, :].T
assert train_x.shape == (15, 1000)
X_normal, X_mean, X_std = normal(train_x.T)
train_x = X_normal.T

test_x = x[1000:, :]
test_x = (test_x - X_mean) / (X_std + 1e-8)
test_x = test_x.T
test_y = y[1000:, :].T
assert test_x.shape == (15, 424)

layers_dims = [15, 10, 10, 10, 1]

parameters3 = L_layer_model(train_x, train_y, layers_dims, learning_rate=0.002567161349434759, mini_batch_size=16,
                            lambd=0.012, num_iterations=50,
                            beta=0.9, optimizer='adam', print_cost=True, isPlot=True, over_stop=False)
print(type(parameters3[0]))
print(parameters3[0].keys())
key = list(parameters3[0].keys())
print(key)

file = open("../data/para.csv", 'r')


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
with codecs.open("../data/para.csv", 'w', encoding='utf-8') as f:
    json.dump(parameters3[0], f, cls=NpEncoder)


# 读取字典
file = open("../data/para.csv", 'r')

js = file.read()
dict_num = json.loads(js)
for i in dict_num.keys():
    dict_num[i] = np.array(dict_num[i])


# def cmp_dict(src_data, dst_data):
#     assert type(src_data) == type(dst_data), "type: '{}' != '{}'".format(type(src_data), type(dst_data))
#     if isinstance(src_data, dict):
#         assert len(src_data) == len(dst_data), "dict len: '{}' != '{}'".format(len(src_data), len(dst_data))
#         for key in src_data:
#             # assert dst_data.has_key(key)
#             cmp_dict(src_data[key], dst_data[key])
#     elif isinstance(src_data, np.ndarray):
#         assert len(src_data) == len(dst_data), "list len: '{}' != '{}'".format(len(src_data), len(dst_data))
#         for src_list, dst_list in zip(sorted(src_data), sorted(dst_data)):
#             cmp_dict(src_list, dst_list)
#     else:
#         assert src_data == dst_data, "value '{}' != '{}'".format(src_data, dst_data)
#
# cmp_dict(dict_num, parameters3[0])

cost, error, testy2 = predict(test_x, test_y, parameters3[0], L=len(layers_dims), lambd=0.012)
print(cost)
print(error)

cost, error, testy2 = predict(test_x, test_y, dict_num, L=len(layers_dims), lambd=0.012)
print(cost)
print(error)
data = np.array(data.iloc[:, 21:])
print(data.shape)
print(np.mean(abs(data), axis=0))
plt.show()
