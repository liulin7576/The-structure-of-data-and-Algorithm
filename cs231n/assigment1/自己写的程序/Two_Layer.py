'''
和吴恩达老师不同，这里的数据是行向量的，即每列表示对应的特征，行表示由多少个数据
'''
import numpy as np
import matplotlib.pyplot as plt


def unpickle(filename):  #解pickle数据
    import pickle
    with open(filename, 'rb') as f:
        dataSet = pickle.load(f, encoding = 'bytes')
    return dataSet

def splitData(dataSet):
    trainSet = dataSet[b'data']
    trainLabel = dataSet[b'labels']
    return trainSet, np.mat(trainLabel).T

def visualSomeImage(trainSet, trainLabel, labelNames):  #可视化部分图形
    trainSet *= 255
    trainLabel = np.array(trainLabel)
    numClass = len(labelNames)
    sample_per_class = 7   #在一张图片里可视化7x10
    for y, cls in enumerate(labelNames):
        idxs = np.flatnonzero(trainLabel == y)
        idxs = np.random.choice(idxs, sample_per_class, replace = False)
        for i, idx in enumerate(idxs):
            plt_idx = i * numClass + y + 1
            plt.subplot(sample_per_class, numClass, plt_idx)
            image = trainSet[idx].reshape(3, 32, 32).transpose(1, 2, 0).astype('float')
            plt.imshow(image.astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()
                

def get_All_Data():  #加载图片数据集，并且划分为训练集和测试集（首先并没有划分dev集）
    trainFile1 = r'F:\数据集\斯坦福大学数据集\cifar-10-python\cifar-10-batches-py\data_batch_1'  #其中之一的训练集
    trainFile2 = r'F:\数据集\斯坦福大学数据集\cifar-10-python\cifar-10-batches-py\data_batch_2'  #其中之一的训练集
    trainFile3 = r'F:\数据集\斯坦福大学数据集\cifar-10-python\cifar-10-batches-py\data_batch_3'  #其中之一的训练集
    trainFile4 = r'F:\数据集\斯坦福大学数据集\cifar-10-python\cifar-10-batches-py\data_batch_4'  #其中之一的训练集
    trainFile5 = r'F:\数据集\斯坦福大学数据集\cifar-10-python\cifar-10-batches-py\data_batch_5'  #其中之一的训练集
    labelFile = r'F:\数据集\斯坦福大学数据集\cifar-10-python\cifar-10-batches-py\batches.meta'  #返回的的是下标的英文名字
    testFile = r'F:\数据集\斯坦福大学数据集\cifar-10-python\cifar-10-batches-py\test_batch'  #测试集合

    dataSet = unpickle(trainFile1)  #首先使用的是一个数据集，看看效果
    trainSet, trainLabel = splitData(dataSet)   #训练集
    dataSet = unpickle(testFile)
    testAllSet, testAllLabel = splitData(dataSet)  #测试集
    
    #随机选取1000张图片作测试，看准确度
    randIndex = np.random.permutation(10000)
    shuffled_X = testAllSet[randIndex, :]
    shuffled_Y = testAllLabel[randIndex, :]
    testSet = shuffled_X[:1000, :]   #随机取1000个数据集作为测试
    testLabel = shuffled_Y[:1000, :]
    
    label = unpickle(labelFile)  
    labelNames = label[b'label_names'] #label的名字,例如'cat'
    #visualSomeImage(trainSet, trainLabel, labelNames)  #可视化

    #normlization
    trainSet, testSet = trainSet.astype(float), testSet.astype(float)
    mean_image = np.mean(trainSet, axis = 0)
    trainSet -= np.mat(mean_image)
    testSet -= mean_image
    
    return trainSet, trainLabel, testSet, testLabel, labelNames
    
def miniBatch(trainSet, trainLabel,  batchSize = 256):  #将训练数据划分为minibatch，然后进行训练
    m = trainSet.shape[0] #训练集的总数
    mini_batches = []  #划分训练集存储的地方

    #先打乱顺序，然后再mini-batch
    randIndex = np.random.permutation(m)
    shuffled_X = trainSet[randIndex, :]
    shuffled_Y = trainLabel[randIndex, :]

    num_complete_minibatches = int(m/batchSize)
    for k in range(num_complete_minibatches):
        minibatch_X = shuffled_X[k*batchSize : (k+1)*batchSize, :]
        minibatch_Y = shuffled_Y[k*batchSize : (k+1)*batchSize, :]
        minibatch = (minibatch_X, minibatch_Y)
        mini_batches.append(minibatch)

    if m % batchSize != 0:  #如果数据不是minibatch的整数倍，对剩下的单独处理
        minibatch_X = shuffled_X[num_complete_minibatches*batchSize:, :]
        minibatch_Y = shuffled_Y[num_complete_minibatches*batchSize:, :]
        mini_batches.append((minibatch_X, minibatch_Y))

    return mini_batches

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



#开始训练,2层神经网络，1层用Relu，output层用softmax
def initial_parameters(layer_dims):  #初始化参数，构建2层神经网络模型
    n_x, n_h, n_y = layer_dims
    W1 = np.random.randn(n_h, n_x) * 0.01       
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01       
    b2 = np.zeros((n_y, 1))


    parameters = {'W1':W1,
                  'b1':b1,
                  'W2':W2,
                  'b2':b2}

    return parameters
    
def forward(X, parameters):  #softmax前向传播
    W1 = parameters['W1']  #加载参数
    b1 = parameters['b1']
    W2 = parameters['W2']  #加载参数
    b2 = parameters['b2']
    
    Z1 = np.dot(W1, X.T) + b1
    A1 = np.maximum(0, Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)


    cache = (A1, W1, Z1, A2, W2, Z2)  #保存这些值便于backprop计算
    return A2, cache

def compute_loss(A, Y, parameters, regRate):  #计算loss
    m = Y.shape[0]
    
    dataLoss = -np.sum(np.log(A[Y.T, np.arange(m)])) / m
    
    regLoss = regRate*np.sum(np.square(parameters['W1'])) + regRate*np.sum(np.square(parameters['W2']))
    totalLoss = dataLoss + regLoss
    return totalLoss

def backprop(X, A2, Y, cache, parameters, regRate):  #计算反向传播
    m = Y.shape[0]
    dZ2 = np.mat(np.zeros((10, m)))  #10表示有多少类
    A2[Y.T, np.arange(m)] -= 1  #将对应的元素-1，即完成了梯度计算
    dZ2 = A2[:]   #复制即可
    #dZ2 = softmax_backward(A2, Y)   
    A1, W1, Z1, A2, W2, Z2 = cache
    
    dW2 = np.dot(dZ2, A1.T)/m  + 2*regRate*W2
    db2 = np.sum(dZ2, axis = 1, keepdims = True)/m  #用keepdims防止假维度
    dA1 = np.dot(W2.T, dZ2)

    dZ1 = np.multiply(dA1, A1 > 0)
    dW1 = np.dot(dZ1, X)/m + 2*regRate*W1
    db1 = np.sum(dZ1, axis = 1, keepdims = True)/m
    
    
    grads = {'dW1':dW1,
             'db1':db1,
             'dW2':dW2,
             'db2':db2}
    return grads

def updatePara(parameters, grads, learning_rate):  #更新参数
    parameters['W1'] = parameters['W1'] - learning_rate * grads['dW1']
    parameters['b1'] = parameters['b1'] - learning_rate * grads['db1']
    parameters['W2'] = parameters['W2'] - learning_rate * grads['dW2']
    parameters['b2'] = parameters['b2'] - learning_rate * grads['db2']

    return parameters



def model(trainSet, trainLabel, layer_dims, learning_rate = 0.001, regRate = 0, batchSize = 256,
          num_epoches = 2000):  #总的模型
    costs = []  #将cost结果存储起来,后面画图使用
    parameters = initial_parameters(layer_dims)
    
    for i in range(num_epoches):
        mini_batches = miniBatch(trainSet, trainLabel,  batchSize = 256)  #构造mini_batch
        for minibatch in mini_batches:
            #选择一个minibatch
            minibatch_X, minibatch_Y = minibatch
            #前向传播
            A2, cache = forward(minibatch_X, parameters)
            #计算成本
            loss = compute_loss(A2, minibatch_Y, parameters, regRate)
            #反向传播
            grads = backprop(minibatch_X, A2, minibatch_Y, cache, parameters, regRate)
            #更新参数
            parameters = updatePara(parameters, grads, learning_rate)

        if i % 200 ==0:
            print('cost after epoch %d: %f' %(i, loss))
    
        if i % 50 ==0:
            costs.append(loss)

    #画出cost图像
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epoch(per 50)')
    plt.title('learning_rate = '+ str(learning_rate))
    plt.show()
    
    
    return parameters

def pred(testSet, testLabel, parameters = None):  #在测试集上跑
    import pickle
    f = open(r'E:\python编程处\cs231n的作业\assigment1\storeParameters_TwoLayers.txt', 'rb')
    parameters = pickle.load(f)
    A2, cache = forward(testSet, parameters)
    maxCol = np.max(A2, axis = 0).T
    m = testLabel.shape[0]
    yhat = np.zeros((m, 1))
    for i in range(m):
        ind = np.nonzero(A2[:, i] == float(maxCol[i]))[0]
        yhat[i] = ind
    return yhat

def storeParameters(parameters):  #存储参数
    import pickle
    f = open(r'E:\python编程处\cs231n的作业\assigment1\storeParameters_TwoLayers.txt', 'wb')
    parameters = pickle.dump(parameters, f)
    f.close()



trainSet, trainLabel, testSet, testLabel, labelNames = get_All_Data()
mini_test, mini_test_label =  trainSet[-200:], trainLabel[-200:]  #选取小的验证集，用来调参
m, n = trainSet.shape
layer_dims = (n, 30, 10)

#parameters = model(trainSet, trainLabel, layer_dims, learning_rate = 0.000075, regRate = 0.8930, batchSize = 256,\
#                   num_epoches = 400)
'''
for circle in range(10): #多次循环选取合适的参数
    learning_rate = 10 ** np.random.uniform(-3, -5)
    regRate = 10 ** np.random.uniform(-4, 0)
    #训练
    parameters = model(trainSet[:-200], trainLabel[:-200], layer_dims, learning_rate, regRate, batchSize = 256,\
                    num_epoches = 1800)


    yhat = pred(trainSet[-200:], trainLabel[-200:], parameters)
    print('cross validation\'s accuracy is %f, lr = %f, reg = %f' %(np.sum(yhat == trainLabel[-200:])/len(trainLabel[-200:]),
          learning_rate, regRate))
'''
#storeParameters(parameters)
yhat = pred(trainSet, trainLabel)
print('train accuracy is %f' %(np.sum(yhat == trainLabel)/len(trainLabel)))
yhat = pred(testSet, testLabel)
print('test accuracy is',np.sum(yhat == testLabel)/len(testLabel))


'''   #验证反向传播的正确性
yhat = np.mat([[0.9, 0.8, 0.7],
        [0.1, 0.2, 0.3]])
Y = np.mat([0, 0, 1]).T
cache = ([[1, 2],
          [1, 2],
          [3, 4]], [[0.1, 0.2],
                    [0.3, 0.5]], 1)
grads = backprop(yhat, Y, cache)
print(grads)
'''



#测试上述代码有没有错误
'''
trainSet, trainLabel, testSet, testLabel = get_All_Data()
mini_batches = miniBatch(trainSet, trainLabel,  batchSize = 256)
'''  





    








