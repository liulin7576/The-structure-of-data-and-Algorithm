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
    testSet, testLabel = splitData(dataSet)  #测试集
    
    #随机选取1000张图片作测试，看准确度
    '''
    randIndex = np.random.permutation(10000)
    shuffled_X = testAllSet[randIndex, :]
    shuffled_Y = testAllLabel[randIndex, :]
    testSet = shuffled_X[:1000, :]   #随机取1000个数据集作为测试
    testLabel = shuffled_Y[:1000, :]
    '''
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
    A, W, Z = cache
    
    dZ = np.multiply(dA, np.int64(A > 0))
    return dZ

def softmax(Z):  #softmax函数的实现
    A = np.exp(Z) / np.sum(np.exp(Z), axis = 0)
    return A

def softmax_backward(AL, Y):  #计算反向传播的softmax微分
    m = Y.shape[0]
    
    AL[Y.T, np.arange(m)] -= 1
    dZL = AL.copy()
    return dZL



#开始训练,L层神经网络
def initial_parameters(layer_dims):  #初始化参数，构建2层神经网络模型
    num_layer = len(layer_dims)
    parameters, v, s = {}, {}, {}    #参数初始化,加入Adam
    for i in range(1, num_layer):

        v['W'+str(i)] = np.zeros((layer_dims[i], layer_dims[i-1]))
        v['b'+str(i)] = np.zeros((layer_dims[i], 1))
        s['W'+str(i)] = np.zeros((layer_dims[i], layer_dims[i-1]))
        s['b'+str(i)] = np.zeros((layer_dims[i], 1))
        
        #parameters['W'+str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.001  #多层神经网络的初始化方式
        
        parameters['W'+str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1])/np.sqrt(layer_dims[i-1]/2) * 0.01  #多层神经网络的初始化方式
        parameters['b'+str(i)] = np.zeros((layer_dims[i], 1))

    return parameters, v, s


    
def forward(X, parameters):  #softmax前向传播
    L = len(parameters) // 2 #神经网络的层数
    caches = []   #供反向传播使用
    A = X.T
    for i in range(1, L):
        A_prev = A
        Z = np.dot(parameters['W'+str(i)], A_prev) + parameters['b'+str(i)]
        A = Relu(Z)
        caches.append((A_prev, parameters['W'+str(i)], Z))

    ZL = np.dot(parameters['W'+str(L)], A) + parameters['b'+str(L)]
    AL = softmax(ZL)

    caches.append((A, parameters['W'+str(L)], ZL))
    return AL, caches

def compute_loss(AL, Y, parameters, regRate):  #计算loss
    m = Y.shape[0]
    
    dataLoss = -np.sum(np.log(AL[Y.T, np.arange(m)])) / m  #计算数据cost

    L = len(parameters)//2
    regLoss = 0
    for i in range(1, L+1):  #计算正则化cost
        regLoss += np.sum(np.square(parameters['W'+str(i)])) * regRate
        
    totalLoss = dataLoss + regLoss  #总的cost
    return totalLoss

def backprop(X, AL, Y, caches, regRate):  #计算反向传播
    m = Y.shape[0]

    dZ = softmax_backward(AL, Y)
    grads = {}   #存储梯度
    L = len(caches)
    for i in range(L, 0, -1):
        A, W, Z = caches[i-1]
        grads['dW'+str(i)] = np.dot(dZ, A.T)/m + 2*regRate*W
        grads['db'+str(i)] = np.sum(dZ, axis = 1, keepdims = True)/m
        dA = np.dot(W.T, dZ)
        dZ = relu_backward(dA, (A, W, Z))

    

    return grads

def updatePara(parameters, grads, learning_rate, v, s, t):  #更新参数
    L = len(parameters)//2
    t = t + 1  #防止t开始为0
    #使用Adam
    belta, belta2, epsilon = 0.9, 0.99, 1e-8
    v_corr, s_corr = {}, {}
    for i in range(1, L+1):
        
        v['W'+str(i)] = belta * v['W'+str(i)] + (1-belta)*grads['dW'+str(i)]  #Momentum
        v['b'+str(i)] = belta * v['b'+str(i)] + (1-belta)*grads['db'+str(i)]  #Momentum
        v_corr['W'+str(i)] = v['W'+str(i)]/(1 - np.power(belta, t))   #修正
        v_corr['b'+str(i)] = v['b'+str(i)]/(1 - np.power(belta, t))
        s['W'+str(i)] = belta2 * s['W'+str(i)] + (1-belta2)*(grads['dW'+str(i)] ** 2)  #RMSProp
        s['b'+str(i)] = belta2 * s['b'+str(i)] + (1-belta2)*(grads['db'+str(i)] ** 2)  #RMSProp
        s_corr['W'+str(i)] = s['W'+str(i)]/(1 - np.power(belta2, t))  #修正
        s_corr['b'+str(i)] = s['b'+str(i)]/(1 - np.power(belta2, t))

        
        new_learning_rate = learning_rate# * (0.95 ** t)  #学习率衰减
        
        parameters['W'+str(i)] = parameters['W'+str(i)] - new_learning_rate * v_corr['W'+str(i)]/(np.sqrt(s_corr['W'+str(i)]) + epsilon)  #Adam更新
        parameters['b'+str(i)] = parameters['b'+str(i)] - new_learning_rate * v_corr['b'+str(i)]/(np.sqrt(s_corr['b'+str(i)]) + epsilon)

    return parameters, v, s



def model(trainSet, trainLabel, layer_dims, learning_rate = 0.05, regRate = 0, batchSize = 256,
          num_epoches = 1000):  #总的模型
    costs = []  #将cost结果存储起来,后面画图使用
    parameters, v, s = initial_parameters(layer_dims)
    
    for i in range(num_epoches):
        mini_batches = miniBatch(trainSet, trainLabel,  batchSize = 256)  #构造mini_batch
        for minibatch in mini_batches:
            #选择一个minibatch
            minibatch_X, minibatch_Y = minibatch
            #前向传播
            AL, caches = forward(minibatch_X, parameters)
            #计算成本
            loss = compute_loss(AL, minibatch_Y, parameters, regRate)
            #反向传播
            grads = backprop(minibatch_X, AL, minibatch_Y, caches, regRate)
            #更新参数
            parameters, v, s = updatePara(parameters, grads, learning_rate, v, s, i)


        if i % 30 ==0:
            print('cost after epoch %d: %f' %(i, loss))
    
        if i % 10 ==0:
            costs.append(loss)
    
    #画出cost图像
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epoch(per 10)')
    plt.title('learning_rate = '+ str(learning_rate))
    plt.show()
    
    
    return parameters

def pred(testSet, testLabel, parameters = None):  #在测试集上跑
    
    import pickle
    f = open(r'E:\python编程处\cs231n的作业\assigment2\storePara_fc_net.txt', 'rb')
    parameters = pickle.load(f)
    
    AL, caches = forward(testSet, parameters)
    
    yhat = np.mat(np.argmax(AL, axis = 0)).reshape(-1, 1)   #求出所有列的最大值对应的项
  
    return yhat

def storeParameters(parameters):  #存储参数
    import pickle
    f = open(r'E:\python编程处\cs231n的作业\assigment2\storePara_fc_net.txt', 'wb')
    parameters = pickle.dump(parameters, f)
    f.close()



trainSet, trainLabel, testSet, testLabel, labelNames = get_All_Data()
mini_test, mini_test_label =  trainSet[:20], trainLabel[:20]  #选取小的验证集，用来验证模型是否正确，正确的话能够过拟合
m, n = trainSet.shape
layer_dims = [n, 100, 30, 10]

parameters = model(trainSet, trainLabel, layer_dims, learning_rate = 0.000466, regRate = 0.02237, batchSize = 256,\
                   num_epoches = 180)
'''
for circle in range(5): #多次循环选取合适的参数
    learning_rate = 10 ** np.random.uniform(-3, -4)
    regRate = 10 ** np.random.uniform(-3, 0)
    #训练
    parameters = model(trainSet, trainLabel, layer_dims, learning_rate, regRate, batchSize = 256,\
                    num_epoches = 200)


    yhat = pred(testSet[:1000], testLabel[:1000], parameters)
    print('cross validation\'s accuracy is %f, lr = %f, reg = %f' %(np.sum(yhat == testLabel[:1000])/len(testLabel[:1000]),
          learning_rate, regRate))
    
'''
storeParameters(parameters)
yhat = pred(trainSet, trainLabel)
print('train accuracy is %f' %(np.sum(yhat == trainLabel)/len(trainLabel)))
yhat = pred(testSet[:200], testLabel)
print('test accuracy is',np.sum(yhat == testLabel[:200])/len(testLabel[:200]))


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





    








