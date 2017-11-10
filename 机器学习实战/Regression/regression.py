import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(filename): #打开数据集并进行处理
    f = open(filename)
    fm = f.readlines()
    trainSet, trainLabel = [], []
    for line in fm:
        lineArr = line.strip().split('\t')
        lineArr = list(map(float, lineArr))
        trainSet.append(lineArr[:-1])
        trainLabel.append(lineArr[-1])
    return np.mat(trainSet), np.mat(trainLabel).T


def standReg(trainSet, trainLabel): #标准的最小二乘回归求解
    XTX = trainSet.T * trainSet
    if np.linalg.det(XTX) == 0:
        print('It is no inverse!')
        return
    wHat = XTX.I * (trainSet.T * trainLabel)
    return wHat

def lwlr(testPoint, trainSet, trainLabel, k = 1.0): #局部加权线性回归方法
    m = trainSet.shape[0]
    weight = np.mat(np.eye(m)) #创建一个对角矩阵且元素为1
    for j in range(m):
        diffMat = testPoint - trainSet[j, :]
        weight[j, j] = np.exp(diffMat * diffMat.T/(-2 * k**2))
    XTX = trainSet.T * weight * trainSet
    if np.linalg.det(XTX) == 0:
        print('No inverse!')
        return
    wHat = XTX.I * (trainSet.T * weight * trainLabel)
    return testPoint * wHat 

def lwlrTest(testArr, trainSet, trainLabel, k = 1.0): #
    m = testArr.shape[0]
    yHat = np.zeros((m, 1))
    for i in range(m):
        yHat[i] = lwlr(testArr[i], trainSet, trainLabel, k)
    return yHat



def lasso(trainSet, trainLabel, eps = 0.01, numIt = 100):   #lasso方法，使用前向逐步回归,效果很好啊
    yMean = trainLabel.mean(0)
    xMean = trainSet.mean(0)
    xVar = trainSet.var(0)
    xMat = (trainSet - xMean)/xVar
    yMat = trainLabel - yMean
    m, n = trainSet.shape
    returnMat = np.mat(np.zeros((numIt, n)))
    ws = np.zeros((n, 1)); wsTest = ws.copy()
    lowestErr = np.inf  #初始化最小误差为无穷大
    for i in range(numIt):
        for j in range(n):
            for sign in [-1, +1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yhat = xMat * wsTest
                rssErr = np.sum(np.square(yMat - yhat))
                if rssErr < lowestErr:
                    lowestErr = rssErr
                    ws = wsTest.copy()
        returnMat[i, :] = ws.T
    return returnMat, xMean, xVar, yMean
                


def ridgeReg(trainSet, trainLabel, lambd = 0.2):  #岭回归
    XTX = (trainSet.T * trainSet + lambd * np.eye(trainSet.shape[1]))
    if np.linalg.det(XTX) == 0:
        print('No inverse')
        return
    wHat = XTX.I * trainSet.T * trainLabel  #这里的求逆矩阵怎么每次都忘了，真的是
    return wHat

def ridgeTest(trainSet, trainLabel):  #测试岭回归
    yMean = np.mean(trainLabel, 0)
    yMat = trainLabel - yMean
    xMean = np.mean(trainSet, 0)
    xVar = np.var(trainSet, 0)
    xMat = (trainSet - xMean)/xVar
    numIter = 30
    wMat = np.zeros((numIter, trainSet.shape[1]))
    for i in range(numIter):
        wHat = ridgeReg(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = wHat.T
    return wMat, yMean, xMean, xVar
        


def plotBestBound(wHat, trainSet, trainLabel):  #画出可视化的图
    n = trainSet.shape[0]
    xcord, ycord = [], []
    for i in range(n):
        xcord.append(trainSet[i, 1])
        ycord.append(float(trainLabel[i]))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(xcord, ycord, c = 'red')
    ax.set_xlim([0, 1.2])
    ax.set_ylim([3, 5])
    y = lwlrTest(trainSet, trainSet, trainLabel, k = 0.01)
    xArr = trainSet.copy()
    srtInd = xArr[:, 1].argsort(0) #需要先排序，不然画图就没有用,这里的0是列排序，1表示行排序
    srtInd = list(map(int, srtInd)) #将矩阵转换为列表，这样就可以直接使用排序的结果了，不然原来的是矩阵，直接使用srtInd就会出错
    ax.plot(xArr[:, 1][srtInd], y[srtInd])
    '''
    xSort = xArr[srtInd][:, 0, :]  #书上的这种做法不是很理解，只能貌似懂
    ySort = y[srtInd][:, 0, :]
    #print(xSort, y)
    #y = float(wHat[1]) * x + float(wHat[0])
    ax.plot(xSort[:, 1], ySort)
    '''
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
        
def predAbalone():
    filename = r'F:\github\MachineLearning全\input\8.Regression\abalone.txt'
    trainSet, trainLabel = loadDataSet(filename)
    '''   #用线性回归预测
    wHat = standReg(trainSet[0:99,:], trainLabel[0:99, :])
    yhat = trainSet[100:199, :] * wHat
    '''
    
    '''   #用加权线性回归做预测
    yhat01 = lwlrTest(trainSet[100:199, :], trainSet[0:99, :], trainLabel[0:99], k = 0.1)
    yhat10 = lwlrTest(trainSet[100:199, :], trainSet[0:99, :], trainLabel[0:99], k = 10)
    yhat1 = lwlrTest(trainSet[100:199, :], trainSet[0:99, :], trainLabel[0:99], k = 1.0)
    err10 = np.sum(np.square(yhat10 - trainLabel[100:199, :]))
    err1 = np.sum(np.square(yhat1 - trainLabel[100:199, :]))
    err01 = np.sum(np.square(yhat01 - trainLabel[100:199, :]))
    print('err10, err1, err01 = ',(err10, err1, err01))
    '''

    '''        # 用岭回归求解误差
    ridgeWeight, yMean, xMean, xVar = ridgeTest(trainSet[0:99, :], trainLabel[0:99, :])
    trainSet[0:199, :] = (trainSet[0:199, :] - xMean)/xVar  #测试的数据也要用相同的均值方差作归一化
    yhat = trainSet[100:199, :] * np.mat(ridgeWeight[15]).T  #只是人工抽取了一个lambd来做测验
    yhat = yhat + yMean #做完预测之后，要转换为原来的数据y，即加上yMean即可
    err = np.sum(np.square(yhat - trainLabel[100:199, :]))  #用trainSet[0:99, :]预测100:199的数据
    print(err)  #计算预测方差，效果很不错，高于前面的线性加权最小回归
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ridgeWeight)
    plt.show()
    '''

            #用前向逐步回归求解误差
    returnMat, xMean, xVar, yMean = lasso(trainSet[0:99, :], trainLabel[0:99, :],eps = 0.005, numIt = 1000)
    trainSet[0:299, :] = (trainSet[0:299, :] - xMean)/xVar
    yhat = (trainSet[100:199, :] * returnMat[20, :].T) + yMean
    err = np.sum(np.square(yhat - trainLabel[100:199]))
    print(returnMat)
    print('error rate is %.3f'%err)  
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(returnMat)
    plt.show()


    

predAbalone()

    
#wHat = standReg(trainSet, trainLabel)
#plotBestBound(wHat, trainSet, trainLabel)
        
