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
    
    #随机选取500张图片作测试，看准确度
    randIndex = np.random.permutation(10000)
    shuffled_X = testAllSet[randIndex, :]
    shuffled_Y = testAllLabel[randIndex, :]
    testSet = shuffled_X[:1000, :]   #随机取1000个数据集作为测试
    testLabel = shuffled_Y[:1000, :]
    
    label = unpickle(labelFile)  
    labelNames = label[b'label_names'] #label的名字。例如'cat'
    #visualSomeImage(trainSet, trainLabel, labelNames)  #可视化
    return trainSet/255, trainLabel, testSet/255, testLabel
    
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

#开始训练
def initial_parameters(n_x, n_y):  #初始化参数，首先构建softmax，没有hidden layer
    W = np.random.randn(n_y, n_x) * 0.001           
    b = np.zeros((n_y, 1))

    parameters = {'W':W,
                  'b':b}

    return parameters
    
def forward(A, parameters):  #softmax前向传播
    W = parameters['W']
    b = parameters['b']
    Z = np.dot(W, A.T) + b

    cache = (A, W, Z)  #保存这些值便于backprop计算
    return Z, cache

def compute_loss(Z, Y, parameters, regRate):  #计算loss
    m = Y.shape[0]
    loss = Z - Z[Y.T, np.arange(m)] + 1  #这里的向量化实现，很强，当然我没想出来
    Bool = loss > 0
    loss = np.multiply(loss, Bool)
    dataLoss = (np.sum(loss) - m)/m
    
    W = parameters['W'] #计算正则化Loss
    regLoss = np.sum(np.square(W)) * regRate  #正则化
    totalLoss = regLoss + dataLoss
    return totalLoss, Bool  #返回loss的原因是：称为jacobian矩阵，后面backprop可以直接使用

def backprop(yhat, Y, cache, parameters, regRate, Bool):  #计算反向传播
    m = Y.shape[0]
    Bool = Bool * np.ones(Bool.shape)  #这里先对Bool进行处理，才能应用
    Bool[Y.T, np.arange(m)] = -(np.sum(Bool, axis = 0) - 1)  #这里要对Bool的对应正确的标签进行处理，即这一列的均值-1

    A_prev, W, Z = cache
    dW = np.dot(Bool, A_prev)/m + regRate * parameters['W'] #直接使用Bool就可以了
    db = 0
    grads = {'dW':dW,
             'db':db}
    return grads

def updatePara(parameters, grads, learning_rate):  #更新参数
    parameters['W'] = parameters['W'] - learning_rate * grads['dW']
    parameters['b'] = parameters['b'] - learning_rate * grads['db']
    return parameters



def model(trainSet, trainLabel, layer_dims, learning_rate = 0.001, regRate = 0.000005, batchSize = 256,
          num_epoches = 2000):  #总的模型
    costs = []  #将cost结果存储起来,后面画图使用
    n_x, n_y = layer_dims
    parameters = initial_parameters(n_x, n_y)
    for i in range(num_epoches):
        mini_batches = miniBatch(trainSet, trainLabel,  batchSize)
        for minibatch in mini_batches:
            #选择一个minibatch
            minibatch_X, minibatch_Y = minibatch
            #前向传播
            Z, cache = forward(minibatch_X, parameters)
            #计算成本
            loss, Bool = compute_loss(Z, minibatch_Y, parameters, regRate)
            #反向传播
            grads = backprop(Z, minibatch_Y, cache, parameters, regRate, Bool)
            #更新参数
            parameters = updatePara(parameters, grads, learning_rate)

        if i % 100 ==0:
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

def pred(testSet, testLabel):  #在测试集上跑
    import pickle
    f = open(r'E:\python编程处\cs231n的作业\storeParameters_Svm.txt', 'rb')
    parameters = pickle.load(f)
    Z = np.dot(parameters['W'], testSet.T) + parameters['b']
    maxCol = np.max(Z, axis = 0).T
    m = testLabel.shape[0]
    yhat = np.zeros((m, 1))
    for i in range(m):
        ind = np.nonzero(Z[:, i] == float(maxCol[i]))[0]
        yhat[i] = ind
    return yhat

def storeParameters(parameters):  #存储参数
    import pickle
    f = open(r'E:\python编程处\cs231n的作业\storeParameters_Svm.txt', 'wb')
    parameters = pickle.dump(parameters, f)
    f.close()



trainSet, trainLabel, testSet, testLabel = get_All_Data()
m, n = trainSet.shape
layer_dims = (n, 10)

#训练
parameters = model(trainSet, trainLabel, layer_dims, learning_rate = 0.001, batchSize = 256,
          num_epoches = 2000)

storeParameters(parameters)
yhat = pred(trainSet, trainLabel)
print('train accuracy is',np.sum(yhat == trainLabel)/len(trainLabel))
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





    








