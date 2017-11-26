'''
和吴恩达老师不同，这里的数据是行向量的，即每列表示对应的特征，行表示由多少个数据
'''
import numpy as np


def Relu(Z):  #定义一个Relu函数
    A = np.maximum(0, Z)   #max(0, Z)
    return A  

def relu_backward(dA, cache):  #计算反向传播的微分
    A1, W1, Z1, A2, W2, Z2 = cache
    
    dZ = np.multiply(dA, np.int64(A1 > 0))
    return dZ

def softmax(Z):  #softmax函数的实现
    A = np.exp(Z) / np.sum(np.exp(Z), axis = 0)
    return A

def softmax_backward(A, Y):  #计算反向传播的softmax微分
    m = Y.shape[0]
    
    A[Y.T, np.arange(m)] -= 1
    dZ2 = A.copy()
    return dZ2


    
def forward(X, parameters):  #softmax前向传播
    W1 = parameters['W1']  #加载参数
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    Z1 = np.dot(W1, X.T) + b1
    A1 = Relu(Z1)  #


    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    cache = (A1, W1, Z1, A2, W2, Z2)  #保存这些值便于backprop计算
    return A2, cache

def compute_loss(A, Y, parameters, regRate):  #计算loss
    m = Y.shape[0]
    
    dataLoss = -np.sum(np.log(A[Y.T, np.arange(m)])) / m
    
    W1 = parameters['W1'] #计算正则化Loss
    W2 = parameters['W2']
    regLoss = np.sum(np.square(W1)) + np.sum(np.square(W2)) * regRate  #正则化
    
    totalLoss = regLoss + dataLoss
    return totalLoss

def backprop(X, A2, Y, cache, parameters, regRate):  #计算反向传播
    m = Y.shape[0]
    dZ2 = np.mat(np.zeros((10, m)))  #10表示有多少类
    A2[Y.T, np.arange(m)] -= 1  #将对应的元素-1，即完成了梯度计算
    dZ2 = A2[:]   #复制即可
    #dZ2 = softmax_backward(A2, Y)   
    A1, W1, Z1, A2, W2, Z2 = cache
    
    dW2 = np.dot(dZ2, A1.T)/m + regRate * W2
    db2 = np.sum(dZ2, axis = 1, keepdims = True)/m  #用keepdims防止假维度
    dA1 = np.dot(W2.T, dZ2)
    #dZ1 = relu_backward(dA1, cache)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = np.dot(dZ1, X)/m + regRate * W1
    db1 = np.sum(dZ1, axis = 1, keepdims = True)/m
    
    
    grads = {'dW1':dW1,
             'db1':db1,
             'dW2':dW2,
             'db2':db2}
    return grads



def model(X, Y,  parameters, learning_rate = 0.001, regRate = 0): #总的模型
                

            #前向传播
    A2, cache = forward(X, parameters)
            #计算成本
    loss = compute_loss(A2, Y, parameters, regRate)
            #反向传播
    grads = backprop(X, A2, Y, cache, parameters, regRate)


    return grads


def storeParameters(parameters):  #存储参数
    import pickle
    f = open(r'E:\python编程处\cs231n的作业\store_Grads.txt', 'wb')
    parameters = pickle.dump(parameters, f)
    f.close()

def gradient_check_n_test_case(): 
    np.random.seed(1)
    x = np.random.randn(3,4)
    y = np.array([1, 2, 0])
    W1 = np.random.randn(5,4) 
    b1 = np.random.randn(5,1) 
    W2 = np.random.randn(3,5) 
    b2 = np.random.randn(3,1) 
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,}
    return x, y, parameters

x, y, parameters = gradient_check_n_test_case()

#训练
grads = model(x, y, parameters, learning_rate = 0.01)
print(grads, parameters)
storeParameters((grads, x, y, parameters))





