
from hyperopt import fmin, tpe, hp, partial,Trials
import hyperopt
import numpy as np
from L_layer_model import *
from predict import *
import matplotlib.pyplot as plt
import math
import sys
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
def normal(X):
    X_mean=np.mean(X,axis=0)
    X_std=np.std(X,axis=0)
    X_normal=(X-X_mean)/(X_std+1e-8)
    return X_normal,X_mean,X_std
dataset = np.loadtxt('test71.csv', delimiter=',')
X1=dataset[0:14000,0:65]   #800x61
#调整信噪比
X1[0:14000,60:61]=dataset[0:14000,70:71]
#X1=np.hstack((X1,dataset[0:400,65:]))
Y1=dataset[0:14000,65:66]    #800x5
X_normal,X_mean,X_std=normal(X1)

train_x=X_normal.T # 61x800
train_y=Y1.T # 5x800

# import hyperopt.pyll.stochastic
# print(hyperopt.pyll.stochastic.sample(space))

# 自定义MLP参数空间
# hidden_layer_sizes=[(10),(50),(100)]
hidden_layer_sizes=[(6),(8),(9),(10),(11),(12),(13),(14),(15),(16),(17),(18),(46)]
#介质厚度已知
hidden_layer_sizes=[(6),(8),(9),(10),(11),(12),(13),(14),(15),(16),(17),(18),(49)]
solver=[('gd'),('momentum'),('RMSprop'),('adam')]
mini_batch_size=[(32),(64),(128),(256)]
# learning_rate=['invscaling','adaptive']

mlp_space = {"hidden_layer_sizes": hp.choice("hidden_layer_sizes", hidden_layer_sizes),
         "solver": hp.choice("solver", solver),
        "learning_rate": hp.uniform("learning_rate",1e-6,1e-2),
        "mini_batch_size":hp.choice("mini_batch_size",mini_batch_size)}

def argsDict_tranform_mlp(argsDict, isPrint=False):
    argsDict["hidden_layer_sizes"] = argsDict["hidden_layer_sizes"]
    argsDict["solver"] = argsDict["solver"]
    argsDict["learning_rate"] = argsDict["learning_rate"]
    argsDict["mini_batch_size"]=argsDict["mini_batch_size"]
    return argsDict


def mlp_factory(argsDict):
    argsDict = argsDict_tranform_mlp(argsDict)
    layers_dims=[train_x.shape[0],argsDict["hidden_layer_sizes"],train_y.shape[0]]


# 训练以及损失计算
    parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate=argsDict["learning_rate"], mini_batch_size=argsDict["mini_batch_size"], lambd=0,
                                num_iterations=1000,
                                beta=0.9, optimizer=argsDict["solver"], print_cost=False, isPlot=False)
    probas, caches = L_model_forward(train_x, parameters[0])
    cost = compute_cost(probas, train_y, parameters[0], len(layers_dims), lambd=0)
    if math.isnan(cost):
        cost = sys.float_info.max
    return cost

trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=1)
# algo=hyperopt.rand.suggest
best = fmin(mlp_factory, mlp_space, algo=algo, max_evals=20,trials=trials)
#查看其他抽样
# for trial in trials.trials[:]:
#     print(trial['result']['loss'])
#     print(trial['misc']['vals'])

argsDict = argsDict_tranform_mlp(best)
print('hidden_layer_sizes:',hidden_layer_sizes[argsDict["hidden_layer_sizes"]],'learning_rate:',argsDict["learning_rate"],'mini_batch_size:',mini_batch_size[argsDict["mini_batch_size"]],'solver:',
      solver[argsDict["solver"]])
# layers_dims=[train_x.shape[0],hidden_layer_sizes[argsDict["hidden_layer_sizes"]],train_y.shape[0]]
# parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate=argsDict["learning_rate"],
#                            mini_batch_size=mini_batch_size[argsDict["mini_batch_size"]], lambd=0,
#                            num_iterations=1000,
#                            beta=0.9, optimizer=solver[argsDict["solver"]], print_cost=True, isPlot=True)
# x2=dataset[14000:20000,0:65]
# x2[0:6000,60:61]=dataset[14000:20000,68:69]
# y2=dataset[14000:20000,65:66]
# x2=(x2-X_mean)/(X_std+1e-8)
# test_x=x2.T # 61x800
# test_y=y2.T # 5x800
# cost,error,testy1 = predict(train_x, train_y, parameters[0],L=len(layers_dims),lambd=0)
# print(cost)
# print(error)
# cost,error,testy2 = predict(test_x, test_y, parameters[0],L=len(layers_dims),lambd=0)
# print(cost)
# print(error)
# plt.show()


