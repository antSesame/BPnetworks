import numpy as np
from L_layer_model import *
import matplotlib.pyplot as plt
from predict import *
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
def opt_iter_compare(train_x,train_y,hidden_layer_sizes,learning_rate,mini_batch_size,test_x,test_y):
    layers_dims=[train_x.shape[0],hidden_layer_sizes,train_y.shape[0]]
    parameters3 = L_layer_model(train_x, train_y, layers_dims, learning_rate=learning_rate, mini_batch_size=mini_batch_size,
                                lambd=0, num_iterations=1000,
                                beta=0.9, optimizer='gd', print_cost=True, isPlot=True, over_stop=True)
    parameters2 = L_layer_model(train_x, train_y, layers_dims, learning_rate=learning_rate, mini_batch_size=mini_batch_size,
                                lambd=0, num_iterations=1000,
                                beta=0.9, optimizer='momentum', print_cost=True, isPlot=True, over_stop=True)
    parameters1 = L_layer_model(train_x, train_y,layers_dims, learning_rate=learning_rate, mini_batch_size=mini_batch_size,
                                lambd=0, num_iterations=1000,
                                beta=0.9, optimizer='RMSprop', print_cost=True, isPlot=True, over_stop=True)
    parameters0 = L_layer_model(train_x, train_y, layers_dims, learning_rate=learning_rate, mini_batch_size=mini_batch_size,
                                lambd=0, num_iterations=1000,
                                beta=0.9, optimizer='adam', print_cost=True, isPlot=True, over_stop=True)

    cost, error1, testy2 = predict(train_x, train_y, parameters3[0], L=len(layers_dims), lambd=0)
    cost, error2, testy2 = predict(test_x, test_y, parameters3[0], L=len(layers_dims), lambd=0)
    print(error1,error2)
    cost, error1, testy2 = predict(train_x, train_y, parameters2[0], L=len(layers_dims), lambd=0)
    cost, error2, testy2 = predict(test_x, test_y, parameters2[0], L=len(layers_dims), lambd=0)
    print(error1, error2)
    cost, error1, testy2 = predict(train_x, train_y, parameters1[0], L=len(layers_dims), lambd=0)
    cost, error2, testy2 = predict(test_x, test_y, parameters1[0], L=len(layers_dims), lambd=0)
    print(error1, error2)
    cost, error1, testy2 = predict(train_x, train_y, parameters0[0], L=len(layers_dims), lambd=0)
    cost, error2, testy2 = predict(test_x, test_y, parameters0[0], L=len(layers_dims), lambd=0)
    print(error1, error2)

    print(len(parameters3[1])-11,len(parameters2[1])-11,len(parameters1[1])-11,len(parameters0[1])-11)

def normal(X):
    X_mean=np.mean(X,axis=0)
    X_std=np.std(X,axis=0)
    X_normal=(X-X_mean)/(X_std+1e-8)
    return X_normal,X_mean,X_std
def opt_iter_compare_impl():
    dataset = np.loadtxt('test71.csv', delimiter=',')
    X1 = dataset[0:14000, 0:61]  # 800x61
    # 10dB
    X1[0:14000, 60:61] = dataset[0:14000, 70:71]
    # X1=np.hstack((X1,dataset[0:400,65:]))
    Y1 = dataset[0:14000, 65:66]  # 800x5
    X_normal, X_mean, X_std = normal(X1)
    train_x = X_normal.T  # 61x800
    train_y = Y1.T  # 5x800

    x2 = dataset[14000:20000, 0:61]
    x2[0:6000, 60:61] = dataset[14000:20000, 70:71]
    y2 = dataset[14000:20000, 65:66]
    x2 = (x2 - X_mean) / (X_std + 1e-8)
    test_x = x2.T  # 61x800
    test_y = y2.T  # 5x800
    opt_iter_compare(train_x, train_y, 8, 0.0007112048067781445, 32, test_x, test_y)
    plt.show()
def dist_error_get(test_x,test_y,parameters,L):
    a = []
    b = []
    d = []
    e = []
    f = []
    g = []
    h = []
    p = []
    q = []
    cost, error, testy2 = predict(test_x, test_y, parameters, L=L, lambd=0)
    pred_y = np.squeeze(testy2)
    true_y = np.squeeze(test_y)
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
    return a
'''
获取训练集和验证集
输入1，代表信噪比为10.输入为2，代表信噪比为20，以此类推
'''
def get_Xy(i):
    dataset = np.loadtxt('test71.csv', delimiter=',')
    X1 = dataset[0:14000, 0:61]  # 800x61
    # 10dB
    X1[0:14000, 60:61] = dataset[0:14000, 66+i:67+i]
    # X1=np.hstack((X1,dataset[0:400,65:]))
    Y1 = dataset[0:14000, 65:66]  # 800x5
    X_normal, X_mean, X_std = normal(X1)
    train_x = X_normal.T  # 61x800
    train_y = Y1.T  # 5x800
    x2 = dataset[14000:20000, 0:61]
    x2[0:6000, 60:61] = dataset[14000:20000, 66+i:67+i]
    y2 = dataset[14000:20000, 65:66]
    x2 = (x2 - X_mean) / (X_std + 1e-8)
    test_x = x2.T  # 61x800
    test_y = y2.T  # 5x800
    return train_x,train_y,test_x,test_y

def dist_snr_compare():
    train_x, train_y, test_x, test_y=get_Xy(1)
    layers_dims = [61, 9, 1]
    parameters3 = L_layer_model(train_x, train_y, layers_dims, learning_rate=0.002567161349434759,
                                mini_batch_size=64,
                                lambd=0, num_iterations=1000,
                                beta=0.9, optimizer='adam', print_cost=True, isPlot=True, over_stop=False)
    a=dist_error_get(test_x,test_y,parameters3[0],3)
    train_x, train_y, test_x, test_y = get_Xy(2)
    layers_dims = [61, 9, 1]
    parameters3 = L_layer_model(train_x, train_y, layers_dims, learning_rate=0.008882819966208407,
                                mini_batch_size=128,
                                lambd=0, num_iterations=1000,
                                beta=0.9, optimizer='adam', print_cost=True, isPlot=True, over_stop=False)
    a1 = dist_error_get(test_x, test_y, parameters3[0], 3)
    train_x, train_y, test_x, test_y = get_Xy(3)
    layers_dims = [61, 12, 1]
    parameters3 = L_layer_model(train_x, train_y, layers_dims, learning_rate=0.0018417745354362648,
                                mini_batch_size=32,
                                lambd=0, num_iterations=1000,
                                beta=0.9, optimizer='adam', print_cost=True, isPlot=True, over_stop=False)
    a2 = dist_error_get(test_x, test_y, parameters3[0], 3)
    train_x, train_y, test_x, test_y = get_Xy(4)
    layers_dims = [61, 8, 1]
    parameters3 = L_layer_model(train_x, train_y, layers_dims, learning_rate=0.007112048067781445,
                                mini_batch_size=32,
                                lambd=0, num_iterations=1000,
                                beta=0.9, optimizer='adam', print_cost=True, isPlot=True, over_stop=False)
    a3 = dist_error_get(test_x, test_y, parameters3[0], 3)

    b = ['5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40']
    fig = plt.figure()
    plt.plot(b, a, 'b*-')
    plt.plot(b, a1, 'rd-')
    plt.plot(b, a2, 'ys-')
    plt.plot(b, a3, 'go-')
    plt.ylabel('error/m')
    plt.xlabel('distance/m')
    plt.legend(['10dB','20dB','30dB','40dB'])



    c1 = [a[0], a1[0], a2[0], a3[0]]
    c2 = [a[1], a1[1], a2[1], a3[1]]
    c3 = [a[2], a1[2], a2[2], a3[2]]
    c4 = [a[3], a1[3], a2[3], a3[3]]
    c5 = [a[4], a1[4], a2[4], a3[4]]
    c6 = [a[5], a1[5], a2[5], a3[5]]
    c7 = [a[6], a1[6], a2[6], a3[6]]
    b=['10dB','20dB','30dB','40dB']
    plt.savefig('1.png')
    plt.figure()
    plt.plot(b, c1, 'b*-')
    plt.plot(b, c2, 'rd-')
    plt.plot(b, c3, 'ys-')
    plt.plot(b, c4, 'go-')
    plt.plot(b, c5, 'cx-')
    plt.plot(b, c6, 'kh-')
    plt.plot(b, c7, 'm+-')
    plt.ylabel('error/m')
    plt.xlabel('SNR/dB')
    plt.legend(['5-10m', '10-15m', '15-20m', '20-25m', '25-30m', '30-35m', '35-40m'],loc='upper right')
    plt.savefig('2.png')
    plt.show()


def get_Xy_z(i):
    dataset = np.loadtxt('test71.csv', delimiter=',')
    X1 = dataset[0:14000, 0:65]  # 800x61
    # 10dB
    X1[0:14000, 60:61] = dataset[0:14000, 66+i:67+i]
    # X1=np.hstack((X1,dataset[0:400,65:]))
    Y1 = dataset[0:14000, 65:66]  # 800x5
    X_normal, X_mean, X_std = normal(X1)
    train_x = X_normal.T  # 61x800
    train_y = Y1.T  # 5x800
    x2 = dataset[14000:20000, 0:65]
    x2[0:6000, 60:61] = dataset[14000:20000, 66+i:67+i]
    y2 = dataset[14000:20000, 65:66]
    x2 = (x2 - X_mean) / (X_std + 1e-8)
    test_x = x2.T  # 61x800
    test_y = y2.T  # 5x800
    return train_x,train_y,test_x,test_y
def dist_snr_compare_z():
    train_x, train_y, test_x, test_y=get_Xy_z(1)
    layers_dims = [65, 16, 1]
    parameters3 = L_layer_model(train_x, train_y, layers_dims, learning_rate= 0.00665868165081087,
                                mini_batch_size=32,
                                lambd=0, num_iterations=1000,
                                beta=0.9, optimizer='adam', print_cost=True, isPlot=True, over_stop=False)
    a=dist_error_get(test_x,test_y,parameters3[0],3)
    train_x, train_y, test_x, test_y = get_Xy_z(2)
    layers_dims = [65, 16, 1]
    parameters3 = L_layer_model(train_x, train_y, layers_dims, learning_rate=0.006031883351909358,
                                mini_batch_size=32,
                                lambd=0, num_iterations=1000,
                                beta=0.9, optimizer='adam', print_cost=True, isPlot=True, over_stop=False)
    a1 = dist_error_get(test_x, test_y, parameters3[0], 3)
    train_x, train_y, test_x, test_y = get_Xy_z(3)
    layers_dims = [65, 9, 1]
    parameters3 = L_layer_model(train_x, train_y, layers_dims, learning_rate=0.009915288996292347,
                                mini_batch_size=64,
                                lambd=0, num_iterations=1000,
                                beta=0.9, optimizer='adam', print_cost=True, isPlot=True, over_stop=False)
    a2 = dist_error_get(test_x, test_y, parameters3[0], 3)
    train_x, train_y, test_x, test_y = get_Xy_z(4)
    layers_dims = [65,17, 1]
    parameters3 = L_layer_model(train_x, train_y, layers_dims, learning_rate=0.004275548446662426,
                                mini_batch_size=32,
                                lambd=0, num_iterations=1000,
                                beta=0.9, optimizer='adam', print_cost=True, isPlot=True, over_stop=False)
    a3 = dist_error_get(test_x, test_y, parameters3[0], 3)

    b = ['5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40']
    fig = plt.figure()
    plt.plot(b, a, 'b*-')
    plt.plot(b, a1, 'rd-')
    plt.plot(b, a2, 'ys-')
    plt.plot(b, a3, 'go-')
    plt.ylabel('error/m')
    plt.xlabel('distance/m')
    plt.legend(['10dB','20dB','30dB','40dB'])



    c1 = [a[0], a1[0], a2[0], a3[0]]
    c2 = [a[1], a1[1], a2[1], a3[1]]
    c3 = [a[2], a1[2], a2[2], a3[2]]
    c4 = [a[3], a1[3], a2[3], a3[3]]
    c5 = [a[4], a1[4], a2[4], a3[4]]
    c6 = [a[5], a1[5], a2[5], a3[5]]
    c7 = [a[6], a1[6], a2[6], a3[6]]
    b=['10dB','20dB','30dB','40dB']
    plt.savefig('3.png')
    plt.figure()
    plt.plot(b, c1, 'b*-')
    plt.plot(b, c2, 'rd-')
    plt.plot(b, c3, 'ys-')
    plt.plot(b, c4, 'go-')
    plt.plot(b, c5, 'cx-')
    plt.plot(b, c6, 'kh-')
    plt.plot(b, c7, 'm+-')
    plt.ylabel('error/m')
    plt.xlabel('SNR/dB')
    plt.legend(['5-10m', '10-15m', '15-20m', '20-25m', '25-30m', '30-35m', '35-40m'],loc='upper right')
    plt.savefig('4.png')
    plt.show()
# opt_iter_compare_impl()
dist_snr_compare_z()