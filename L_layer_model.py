import matplotlib.pyplot as plt
from code.init_para import *
from code.Update_Parameters import *
from code.L_model_forward import *
from code.L_model_backward import *
from code.cost_func import *
from code.momentum import *
from code.adam import *
from code.RMSprop import *
from code.mini_batch import *
from code.predict import getError
import time


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=1500,lambd=0,optimizer='gd',mini_batch_size=64,beta=0.9,beta1=0.9,beta2=0.999,
          epsilon=1e-8,print_cost=False, isPlot=True,over_stop=True):
    """
    实现一个L层神经网络：[LINEAR-> RELU] *（L-1） - > LINEAR-> SIGMOID。

    参数：
	    X - 输入的数据，维度为(n_x，例子数)
        Y - 标签，向量，0为非猫，1为猫，维度为(1,数量)
        layers_dims - 层数的向量，维度为(n_y,n_h,···,n_h,n_y)
        learning_rate - 学习率
        num_iterations - 迭代的次数
        print_cost - 是否打印成本值，每100次打印一次
        isPlot - 是否绘制出误差值的图谱

    返回：
     parameters - 模型学习的参数。 然后他们可以用来预测。
    """
    np.random.seed(1)
    costs = []
    errors=[]
    L = len(layers_dims)
    seed = 10 #随机种子
    t = 0  # 每学习完一个minibatch就增加1
    n_iter_no_change = 0



    parameters = initialize_parameters_deep(layers_dims)
    bestPara={}
    bestPara2={}
    if optimizer == "gd":
        pass  # 不使用任何优化器，直接使用梯度下降法
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)  # 使用动量
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)  # 使用Adam优化
    elif optimizer=="RMSprop":
        s=initialize_RMSprop(parameters)
    else:
        print("optimizer参数错误，程序退出。")
        exit(1)
    # seed = seed + 1
    # minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
    start=time.time()
    no_learn=False
    over_learn=False
    iter_over=0
    for i in range(0, num_iterations):
        # 定义随机 minibatches,我们在每次遍历数据集之后增加种子以重新排列数据集，使每次数据的顺序都不同

        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        cost_batchs=[]
        for minibatch in minibatches:
            # 选择一个minibatch
            (minibatch_X, minibatch_Y) = minibatch
            AL, caches = L_model_forward(minibatch_X, parameters)

            cost = compute_cost(AL, minibatch_Y, parameters, L, lambd)
            if math.isnan(cost):
                no_learn=True
                break
            cost_batchs.append(cost)
            grads = L_model_backward(AL, minibatch_Y, caches, lambd)
            # #通过梯度趋于0判断已收敛
            # grad = gradients_to_vector(grads)
            # grad1=np.zeros((len(grad),1))
            # numerator = np.linalg.norm(grad - grad1)  # Step 1'
            # denominator = np.linalg.norm(grad) + np.linalg.norm(grad1)  # Step 2'
            # difference = numerator / denominator
            # if difference < 1e-7:
            #     over_learn = True
            #     break

            if optimizer == "gd":
                parameters = update_parameters(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2,
                                                               epsilon)
            elif optimizer=="RMSprop":
                parameters, s = update_parameters_with_RMSprop(parameters, grads,s, beta, learning_rate,epsilon)


        # 打印成本值，如果print_cost=False则忽略
        #如果梯度趋于0停止学习
        if over_learn==True:
            print(i,'已收敛')
            break
        if i % 1 == 0:
            # 记录成本
            costs.append(np.mean(cost_batchs))
            # 是否打印成本值
            if print_cost and i%1==0:
                print("第", i, "次迭代，成本值为：", np.squeeze(costs[-1]),'最小值',np.min(costs))
        AL, caches = L_model_forward(X, parameters)
        error = getError(AL, Y)
        errors.append(error)

        if over_stop==True:
            if i != 0 and costs[-1] > np.min(costs[:-1])-1e-5:
                n_iter_no_change += 1
            else:
                n_iter_no_change = 0
                bestPara = parameters.copy()
            if n_iter_no_change > 10:
                iter_over = i
                print('第', i, '次迭代时，学习结束')
                break
        else:
            # AL, caches = L_model_forward(X, parameters)
            # error = getError(AL, Y)
            # errors.append(error)
            # if i % 1 == 0 and print_cost == True:
            #     print("第", i, "次迭代，误差值为：", np.squeeze(error), '最小值', np.min(errors))

            if i!=0 and errors[-1] < np.min(errors[:len(errors) - 1]):
                bestPara = parameters.copy()

        if no_learn == True:
             break
        if i>num_iterations-1:
            print('达到最大训练次数')
            break
            #  early_stoping的实现，利用验证集的误差，为了降低方差防止过拟合
            # if error >= np.min(errors[:len(errors) - 1]) - 1e-3:
            #     n_iter_no_change += 1
            # else:
            #     n_iter_no_change = 0
            #
            # if error < np.min(errors[:len(errors) - 1]):
            #     bestPara = parameters.copy()
            #     n_iter_no_change = 0
            # if n_iter_no_change > 10:
            #     print('第', i, '次已收敛')
            #     print(n_iter_no_change)
            #     break

    # 迭代完成，根据条件绘制图
    end=time.time()
    print('time:'+str((end-start)))
    if isPlot:
        # plt.subplot(211)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Learning rate =" + str(learning_rate))
        # plt.ylim((0,5))
        #测厚度
        # plt.ylim((5,10))
        plt.legend(['sgd','momentum','RMSprop','adam'])
        plt.grid(linestyle=":")
        # # plt.show()


        # plt.plot(np.squeeze(errors)[0:-1:10])
        # plt.ylabel('测距误差(m)')
        # plt.xlabel('迭代次数(百次)')
        # plt.ylim((0, 10))
        # plt.legend(['gd', 'momentum', 'adam'])
        # plt.grid(linestyle=":")


        # 厚度已知

    return bestPara,costs,errors


