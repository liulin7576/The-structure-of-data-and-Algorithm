import numpy as np
import matplotlib.pyplot as plt
import random
import time
random.seed(time.time())


def loadDataSet(): #导入数据并进行数据转换
    f = open(r'F:\github\MachineLearning全\input\5.Logistic\TestSet.txt')
    fm = f.readlines()
    dataMat, labelMat = [], []
    index = 0
    for line in fm:
        buff = line.strip().split('\t')
        dataMat.append([1.0, float(buff[0]), float(buff[1])])
        labelMat.append(int(buff[2]))
        index += 1
    return np.array(dataMat), labelMat

def gradDescent(dataMat, labelMat):  # 梯度下降算法
    m, n =dataMat.shape
    W = np.zeros((1, n))
    maxCycle = 10000
    alpha = 0.01
    for i in range(maxCycle):
        S = np.dot(W, dataMat.T) #这里使用的是吴恩达老师深度学习课程的成本函数作为衡量方式
        A = bigfloat(1/(1 + np.exp(-S)))
        dW = np.dot(A - labelMat, dataMat)/len(dataMat)
        W = W - alpha * dW
    return W

def stoGradDescent(dataMat, labelMat, numIter = 150): #改进的随机梯度下降
    m, n =dataMat.shape
    W = np.zeros((1, n))
    for j in range(numIter):
        for i in range(m):
            alpha = 0.01 + 4/(1 + j + i)
            randIndex = int(random.uniform(0, m))
            S = np.dot(W, dataMat[randIndex].T)
            A = 1/(1 + np.exp(-S))
            dW = np.dot(A[0] - labelMat[randIndex], dataMat[randIndex])
            W = W - alpha * dW
    return W

def STD(dataMat, labelMat, numIter): #台大林老师随机梯度下降算法
    m, n  = dataMat.shape
    W = np.zeros((1, n))
    alpha = 0.01
    for j in range(numIter):
        for i in range(m):
            randIndex = int(random.uniform(0, m))
            S = 1/(1 + np.exp(labelMat[randIndex] * np.dot(W, dataMat[randIndex].T)))
            dW = S * (-labelMat[randIndex] * dataMat[randIndex])
            W = W - alpha * dW
    return W
    


def plotBestFit(dataMat, labelMat, W): #画出最佳拟合直线也就是图形化
    xcord1, ycord1 = [], []
    xcord2, ycord2 = [], []
    for i in range(len(dataMat)):
        if labelMat[i] == 1:
            xcord1.append(dataMat[i][1])
            ycord1.append(dataMat[i][2])
        else:
            xcord2.append(dataMat[i][1])
            ycord2.append(dataMat[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
    ax.scatter(xcord2, ycord2, s = 30, c = 'green')
    x = np.arange(-5, 5, 0.1)
    y = (-W[0] - W[1] * x)/W[2]
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def classify(W, inx): #计算输入的样本点属于0 or 1 ?
    testResult = 1/(1 + np.exp(-np.dot(inx, W.T)))
    if testResult > 0.5:
        return 1
    else:
        return 0

def horseTest(): # 病马的预测
    frTrain = open(r'F:\github\MachineLearning全\input\5.Logistic\HorseColicTraining.txt')
    frTest = open(r'F:\github\MachineLearning全\input\5.Logistic\HorseColicTest.txt')
    trainSet, trainLabel = [], []
    testSet, testLabel = [], []
    for line in frTrain.readlines():
        line = line.strip().split('\t')
        lineArr = list(map(float, line)) # 利用容器map快速计算转换为float的值
        trainSet.append(lineArr[:-1])
        trainLabel.append(int(lineArr[-1]))
    #W = gradDescent(np.array(trainSet), trainLabel)#梯度下降
    W = stoGradDescent(np.array(trainSet), trainLabel, 600) #随机梯度下降
    errCount, totalNum = 0, 0
    for line in frTest.readlines():
        totalNum += 1
        line = line.strip().split('\t')
        lineArr = list(map(float, line)) # 利用容器map快速计算转换为float的值
        if classify(W, np.array(lineArr[:-1])) != int(lineArr[-1]):
            errCount += 1
    #print('The total error rate is {}%'.format(errCount/totalNum))
    return errCount/totalNum

def multiTest():#多预测几次取平均值
    err = 0.0
    for i in range(10):
        err += horseTest()
    print('Average error is {}%'.format(err/10))

'''  # 实现两个特征的数据分类可视化情形
dataMat, labelMat = loadDataSet()
W = gradDescent(dataMat, labelMat)
print(W)
plotBestFit(dataMat, labelMat, W[0])
'''





