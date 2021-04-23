import numpy as np
from ..code.L_layer_model import *
import matplotlib.pyplot as plt
from ..code.predict import *

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def normal(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_normal = (X - X_mean) / (X_std + 1e-8)
    return X_normal, X_mean, X_std


def normal2(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_normal = (X - X_min) / (X_max - X_min + 1e-8)
    return X_normal, X_min, X_max


dataset = np.loadtxt('test71.csv', delimiter=',')
X1 = dataset[0:14000, 0:61]  # 800x61
# 10dB
X1[0:14000, 60:61] = dataset[0:14000, 70:71]
# X1=np.hstack((X1,dataset[0:400,65:]))
Y1 = dataset[0:14000, 65:66]  # 800x5
X_normal, X_mean, X_std = normal(X1)
train_x = X_normal.T  # 61x800

train_y = Y1.T  # 5x800
layers_dims = [61, 7, 1]  # 5-layer model
# parameters = L_layer_model(train_x, train_y, layers_dims,learning_rate=0.1, num_iterations =3500, lambd=0.1,print_cost = True,isPlot=False)

# parameters1=L_layer_model(train_x, train_y, layers_dims, learning_rate=0.0225, mini_batch_size=256, lambd=0, num_iterations=3000,
#               beta=0.9, optimizer='gd', print_cost=True, isPlot=True)
# parameters2=L_layer_model(train_x, train_y, layers_dims, learning_rate=0.0225, mini_batch_size=256, lambd=0, num_iterations=3000,
#               beta=0.9, optimizer='momentum', print_cost=True, isPlot=True)0.01745636673760633
parameters3 = L_layer_model(train_x, train_y, layers_dims, learning_rate=0.0002567161349434759, mini_batch_size=64,
                            lambd=0, num_iterations=1000,
                            beta=0.9, optimizer='adam', print_cost=True, isPlot=True, over_stop=False)
# parameters2=L_layer_model(train_x, train_y, layers_dims, learning_rate=0.002567161349434759, mini_batch_size=64, lambd=0, num_iterations=1000,
#               beta=0.9, optimizer='momentum', print_cost=True, isPlot=True,over_stop=True)
# parameters1=L_layer_model(train_x, train_y, layers_dims, learning_rate=0.002567161349434759, mini_batch_size=64, lambd=0, num_iterations=1000,
#               beta=0.9, optimizer='RMSprop', print_cost=True, isPlot=True,over_stop=True)
# parameters0=L_layer_model(train_x, train_y, layers_dims, learning_rate=0.002567161349434759, mini_batch_size=64, lambd=0, num_iterations=1000,
#               beta=0.9, optimizer='adam', print_cost=True, isPlot=True,over_stop=True)


x2 = dataset[14000:20000, 0:61]
x2[0:6000, 60:61] = dataset[14000:20000, 70:71]
y2 = dataset[14000:20000, 65:66]
x2 = (x2 - X_mean) / (X_std + 1e-8)
test_x = x2.T  # 61x800
test_y = y2.T  # 5x800

cost, error, testy2 = predict(train_x, train_y, parameters3[0], L=len(layers_dims), lambd=0)
print(cost)
print(error)
cost, error, testy2 = predict(test_x, test_y, parameters3[0], L=len(layers_dims), lambd=0)
print(cost)
print(error)

# #
# cost,error,testy2 = predict(test_x, test_y, parameters2[0],L=len(layers_dims),lambd=0)
# print(cost)
# print(error)
#
# cost,error,testy2 = predict(test_x, test_y, parameters1[0],L=len(layers_dims),lambd=0)
# print(cost)
# print(error)
# cost,error,testy2 = predict(test_x, test_y, parameters0[0],L=len(layers_dims),lambd=0)
# print(cost)
# print(error)


# 噪声对测距性能的影响（(厚度未知)
# # #信噪比为0db#
# x2[0:6000,60:61]=dataset[14000:20000,66:67]
# # #信噪比为10db#
# x2[0:2000,60:61]=dataset[8000:10000,67:68]
# # #信噪比为20db#
# x2[0:2000,60:61]=dataset[8000:10000,68:69]
# # #信噪比为30db#
# x2[0:2000,60:61]=dataset[8000:10000,69:70]
# # #信噪比为40db#
# x2[0:2000,60:61]=dataset[8000:10000,70:71]
# errors=[]
# for i in range(0,5):
#     x2 = dataset[14000:20000, 0:61]
#     x2[0:6000, 60:61] = dataset[14000:20000, 66+i:67+i]
#     y2 = dataset[14000:20000, 65:66]
#     x2 = (x2 - X_mean) / (X_std + 1e-8)
#     test_x = x2.T  # 61x800
#     test_y = y2.T  # 5x800
#     cost, error, testy2 = predict(test_x, test_y, parameters3, L=len(layers_dims), lambd=0)
#     errors.append(error)
# x=[0,10,20,30,40]
# fig=plt.figure()
# plt.plot(x,errors,'b-')
# plt.ylabel('测距误差(m)')
# plt.xlabel('信噪比(dB)')
# plt.xticks(x)


#
# 距离对测距性能的影响(厚度未知)
a = []
b = []
# x2 = dataset[7000:10000, 0:61]
#
# y2 = dataset[7000:10000, 65:66]
# x2 = (x2 - X_mean) / (X_std + 1e-8)
# test_x = x2.T  # 61x800
# test_y = y2.T  # 5x800
cost, error, testy2 = predict(test_x, test_y, parameters3[0], L=len(layers_dims), lambd=0)
pred_y = np.squeeze(testy2)
true_y = np.squeeze(test_y)
d = []
e = []
f = []
g = []
h = []
p = []
q = []
for i in range(0, 6000):

    if (true_y[i] > 5 and true_y[i] <= 10):
        d.append(abs(pred_y[i] - true_y[i]))
    if (true_y[i] > 10 and true_y[i] <= 15):
        e.append(abs(pred_y[i] - true_y[i]))
    if (true_y[i] > 15 and true_y[i] <= 20):
        f.append(abs(pred_y[i] - true_y[i]))
    if (true_y[i] > 20 and true_y[i] <= 25):
        g.append(abs(pred_y[i] - true_y[i]))
    if (true_y[i] > 25 and true_y[i] <= 30):
        h.append(abs(pred_y[i] - true_y[i]))
    if (true_y[i] > 30 and true_y[i] <= 35):
        p.append(abs(pred_y[i] - true_y[i]))
    if (true_y[i] > 35 and true_y[i] <= 40):
        q.append(abs(pred_y[i] - true_y[i]))
# print(len(d), len(e), len(f), len(g), len(h), len(p), len(q))
d = np.mean(d)
e = np.mean(e)
f = np.mean(f)
g = np.mean(g)
h = np.mean(h)
p = np.mean(p)
q = np.mean(q)
a = [d, e, f, g, h, p, q]
b = ['5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40']
fig = plt.figure()
plt.plot(b, a, 'b-')
plt.plot(b, a, 'b*')
plt.ylabel('测距误差(m)')
plt.xlabel('距离(m)')

# print(a)

plt.show()
