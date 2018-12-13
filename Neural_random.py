#-*- coding:utf-8 -*-
'随机生成数据进入输入层，输出只有一个神经元，进行二分类，训练数据测试数据均随机生成'
'使用随机梯度下降'
'使用Relu作为中间层激活函数,最后一层使用Sigmoid作为判断函数'

import numpy as np

np.random.seed(1)
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def sigmoid_backward(dA, cache):
    '''

    :param dA:post-activation gradient, of any shape
    :param cache: 'Z' where we store for computing backward propagation efficiency
    :return: dZ: Gradient of the cost with respect to Z
    '''

    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert(dZ.shape == Z.shape)
    return dZ

def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape==Z.shape)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) # converting dz to a correct object
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

#随机生成权重值
def weight_variable(shape):
    mean, std = 0, 0.1
    initial = np.random.normal(mean,std,shape)
    return initial
#随机生成偏置值
def bias_variable(shape):
    initial = np.ones(shape) * 0.1
    return initial

#初始化所有神经元参数
def initialize_parameters_deep(layer_dims):
    '''

    :param layer_dims: python array (list) 包含神经网络中神经元的层数及个数
    :return: parameters:
                    字典形式，包括 W1,b1,....WL,bL:
                    WL: L层的权重值(layer_dims[l],layer_dims[l-1]）
                    bL: l层的偏置向量(layer_dims[l],1)
    '''
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) #神经网络的层数

    for l in range(1,L):
        parameters['W' + str(l)] = weight_variable((layer_dims[l], layer_dims[l-1]))
        parameters['b' + str(l)] = bias_variable((layer_dims[l],1))    # assert (y_training.shape[0]==1)
        assert(parameters['W' + str(l)].shape == (layer_dims[l],layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l],1))

    return parameters

def linear_forward(A,W,b):
    '''
    前向传播的线性部分
    :param A:   上一层的输出或来自输入层(size of previous layer, number of examples)
    :param W:   权重向量: 数组，格式(size of current layer, size of previous layer)
    :param b:   偏置向量, 格式(size of current layer,1)
    :return:    Z:  激活函数的输入
                cache:  字典，包含 'A','W','b'; 加快反向传播效率
    '''
    Z = np.dot(W, A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A,W,b)
    return Z,cache

def linear_activation_forward(A_prev, W, b, activation):
    '''
    线性函数到激活函数的传播
    :param A_prev: 上一层的激活函数输出:(size of previous layer, number of examples)
    :param W: 权重向量 (size of current layer, size of previous layer)
    :param b: 偏置向量(size of current layer, 1)
    :param activation: 该层使用的激活函数，字符串:'sigmoid' or 'relu'
    :return: A: 该层激活函数的输出
            cache:  字典，存有 'linear-cache' and 'activation cache';
                    提高反向传播效率
    '''
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    assert(A.shape == (W.shape[0],A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    #linear_cache--->(A,W,b),activation_cache--->Z即激活函数的输入
    return A, cache

def L_model_forward(X, parameters):
    '''
    forward propagation for the [linear->relu]*(L-1)->linear->sigmoid computation
    :param X:神经网络的输入
    :param parameters:字典：存有神经网络中的权重向量、偏置向量
    :return:预测值，以及中间层输出
    '''
    caches = []
    A = X
    L = len(parameters) // 2 # 神经网络的层数

    #使用relu作为激活函数的神经元，在caches末尾添加了上一层的输出cache
    for l in range(1,L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], 'relu')
        caches.append(cache)

    # 传播至最后一层sigmoid激活函数.在caches末尾添加了上一层的输出cache
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)],parameters['b'+str(L)], 'sigmoid')
    caches.append(cache)
    assert(AL.shape == (1,X.shape[1]))
    return AL, caches

def compute_cost(AL, Y):
    '''
    代价函数，没有正则化
    :param AL: 神经网络的预测值, 格式(1, number of examples)
    :param Y: 实际值,shape(1,number of examples)
    :return: 该参数对应下的代价函数值
    '''
    m = Y.shape[1]
    cost = -np.sum(np.multiply(np.log(AL),Y) + np.multiply(np.log(1 - AL), 1 - Y)) / m
    cost = np.squeeze(cost)
    assert(cost.shape==())
    return cost

def linear_backward(dZ, cache):
    '''
    反向传播的线性部分，未经过激活函数
    :param dZ:当前层的代价梯度
    :param cache:元组，存有前一层神经元的输入，该层神经元的权重，偏置
    :return:dA_prev:前一层激活函数输出的代价梯度
            dW:该层权重下的代价梯度
            db:该层偏置下的代价梯度
    '''
    A_prev, W, b = cache
    m = A_prev.shape[1]
    # assert (y_training.shape[0]==1)
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    '''
    线性层--->活化层的反向传播
    :param dA:当前层的前向传播激活函数的梯度
    :param cache:元组，存有（A,W,b),(该层神经元激活函数的输入）
    :param activation:激活函数的类型
    :return:dA_prev:前一层激活函数输出的代价梯度
            dW:该层权重下的代价梯度
            db:该层偏置下的代价梯度
    '''
    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    '''
    反向传播的主函数
    :param AL:神经网络的预测值
    :param Y:真实值
    :param caches:存有中间层的参数
    :return:grads:梯度的字典：
                    grads['dA'+str(l)] = ...
                    grads['dW'+str(l)] = ...
                    grads['db'+str(l)] = ...
    '''
    grads = {}
    L = len(caches) # 神经元的层数
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # 使预测值和真实值维度一致
    # 输出层反向传播梯度计算
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    #最后一层中sigmoid激活函数到线性函数的梯度计算
    current_cache = caches[L-1]
    grads['dA'+str(L)], grads['dW'+str(L)], grads['db'+str(L)] = linear_activation_backward(dAL, current_cache,'sigmoid')

    for l in reversed(range(L-1)):
        # 输出层之前的反向传播
        # 输入: "grads['dA'+str(l+2)],caches" 输出: "grads['dA'+str(l+1)],grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA'+str(l+2)],current_cache,'relu')
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    梯度下降
    """

    L = len(parameters) // 2

    #使用循环，更新权重及偏置向量的值
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iteration = 3000, print_cost = False, SGD=False):
    '''
    L层神经网络的主函数
    :param X:输入值
    :param Y:输入值对应的真实标签
    :param layers_dims:神经网络神经元的参数
    :param learning_rate:学习率
    :param num_iteration:迭代次数
    :param print_cost:是否打印中间参数
    :param SGD:是否开启随机梯度下降
    :return:parameters:神经元的参数，权重及偏置值

    '''

    np.random.seed(1)
    costs = []  # 代价

    # 参数初始化
    parameters = initialize_parameters_deep(layers_dims)


    #循环，计算参数
    for i in range(0, num_iteration):
        #随机梯度下降
        if SGD == 'True':
            #通过随机从输入参数中不放回取500个参数作为神经元的输入，减小计算量
            index = np.random.choice(a=X.shape[0], size=500, replace=False, p=None)
            X = X[index,:]
            Y = Y[index,:]

        # 前向传播
        AL, caches = L_model_forward(X, parameters)

        # 代价计算
        cost = compute_cost(AL, Y)

        # 反向传播
        grads = L_model_backward(AL, Y, caches)

        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)

        # 每100次，进行代价输出
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    return parameters

def gener_dataset(training_shape,test_shape):
    np.random.seed(5)
    X_training = np.random.rand(training_shape[0], training_shape[1])
    y_training = np.hstack((np.ones((1, training_shape[0]//2)), np.zeros((1, training_shape[0]-training_shape[0]//2))))
    # assert (y_training.shape[0]==1)
    np.random.shuffle(y_training)
    X_test = np.random.rand(test_shape[0], test_shape[1])
    y_test = np.hstack((np.ones((1, test_shape[0]//2)), np.zeros((1, test_shape[0]-test_shape[0]//2))))
    np.random.shuffle(y_test)
    return X_training, y_training, X_test, y_test

def predict(X, y, parameters):
    """
    预测函数
    """

    m = X.shape[1]
    n = len(parameters) // 2
    p = np.zeros((1, m))

    # 前向传播
    probas, caches = L_model_forward(X, parameters)

    # 将神经网络输出值转为0-1
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("Accuracy: " + str(np.sum((p == y) / float(m))))

    return p


#生成10000个样本，每个样本维度为50
X_training, y_training, X_test, y_test = gener_dataset([10000,50],[300,50])
#(1000,50),(1000,1),    (300,50),(300,1)
X_training = X_training.T
X_test = X_test.T
layers_dims = [50,50,20,7,5,1]
parameters = L_layer_model(X_training, y_training, layers_dims, num_iteration=2500, print_cost=True, SGD=True)
pred_train = predict(X_training, y_training, parameters)
pred_test = predict(X_test, y_test, parameters)