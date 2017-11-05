'''
用SMO方法来预测马的生死，事实上裕兴速度很慢，因为没有采用核函数，后面采用核函数将会再实验一把
准确率和第五章用Logistic回归差不多，30%多的误差。


'''


import numpy as np
import random
import time
import matplotlib.pyplot as plt
random.seed(time.time())

def loadDataSet(filename): #导入数据并将数据进行分割
    f = open(filename)
    trainSet, trainLabel = [], []

    for line in f.readlines():
        lineArr = line.strip().split('\t')
        trainSet.append(list(map(float, lineArr[:-1])))
        trainLabel.append(float(lineArr[-1]))
    for i in range(len(trainLabel)):
        if int(trainLabel[i]) == 0:
            trainLabel[i] = -1
    return np.mat(trainSet), np.mat(trainLabel).T


def selectJrand(i, m): #随机选取一个除i以外的值
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L): #限定alpha的范围在L和H之间
    if aj > H:
        aj = H
    elif aj < L:
        aj = L
    return aj


class optStruct:  #构造类存储所需的变量，便于访问
    def __init__(self, dataSet, dataLabel, C, toler):
        self.dataSet = dataSet
        self.dataLabel = dataLabel
        self.C = C
        self.toler = toler
        self.m = len(dataSet)
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.cache = np.mat(np.zeros((self.m, 2)))


def calcEk(oS, k): # 计算第k个样本和预测之间的误差
    fxk = np.multiply(oS.alphas, oS.dataLabel).T * \
          (oS.dataSet * oS.dataSet[k, :].T) + oS.b
    Ek = fxk - int(oS.dataLabel[k])
    return Ek

def selectJ(i, oS, Ei): #用于选择第二个alpha值
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.cache[i] = [1, Ei]
    validCache = np.nonzero(oS.cache[:, 0].A)[0] #返回cache中所有第一列非零对应的下标
    if len(validCache) > 1:
        for k in validCache:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            if abs(Ek - Ei) > maxDeltaE:
                maxDeltaE = abs(Ek - Ei)
                Ej = Ek
                maxK = k
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
        return j, Ej

def updateEk(oS, k): # 更新cache中的值
    Ek = calcEk(oS, k)
    oS.cache[k] = [1, Ek] 


def innerL(i, oS): #内层循环计算更新alpha[j]和alpha[i]，如果有更新的话，然后计算更新后的b   
    Ei = calcEk(oS, i)
    if((oS.dataLabel[i] * Ei > oS.toler) and (oS.alphas[i] > 0)) or\
      ((oS.dataLabel[i] * Ei < -oS.toler) and (oS.alphas[i] < oS.C)):
        j, Ej = selectJ(i ,oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if  oS.dataLabel[i] != oS.dataLabel[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            return 0
        eta = 2 * oS.dataSet[i, :] * oS.dataSet[j, :].T - oS.dataSet[i, :] * \
              oS.dataSet[i, :].T - oS.dataSet[j, :] * oS.dataSet[j, :].T
        if eta >= 0:
            return 0
        oS.alphas[j] -= oS.dataLabel[j] * (Ei - Ej)/eta #这里的更新方式一定不能写错了，否则无法正常更新
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if abs(oS.alphas[j] - alphaJold) < 0.0001:
            return 0
        oS.alphas[i] += oS.dataLabel[j] * (oS.dataLabel[i]) * \
                        (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.dataLabel[i] * (oS.alphas[i] - alphaIold) * \
             (oS.dataSet[i, :] * oS.dataSet[i, :].T) -oS.dataLabel[j] * \
             (oS.alphas[j] - alphaJold) * (oS.dataSet[i, :] * oS.dataSet[j, :].T)
        b2 = oS.b - Ej - oS.dataLabel[i] * (oS.alphas[i] - alphaIold) * \
             (oS.dataSet[i, :] * oS.dataSet[j, :].T) -oS.dataLabel[j] * \
             (oS.alphas[j] - alphaJold) * (oS.dataSet[j, :] * oS.dataSet[j, :].T)
        if oS.alphas[i] > 0 and oS.alphas[i] < oS.C:
            oS.b = b1
        elif oS.alphas[j] > 0 and oS.alphas[j] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2
        return 1
    else:
        return 0


def smoP(dataSet, dataLabel, C, toler, maxIter, kTup = ('lin', 0)): #完整的SMO算法过程
    oS = optStruct(dataSet, dataLabel, C, toler)
    iteration = 0
    entireSet, alphaPair = True, 0
    while (iteration < maxIter and alphaPair > 0) or entireSet:
        alphaPair = 0
        if entireSet:
            for i in range(oS.m):
                alphaPair += innerL(i, oS)
            iteration += 1
        else:
            validAlpha = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < oS.C))[0]
            for i in validAlpha:
                alphaPair += innerL(i, oS)
            iteration += 1
        if entireSet:
            entireSet = False
        elif alphaPair == 0:
            entireSet = True
    return oS.b, oS.alphas

def plotBestBound(trainSet, trainLabel, W, b): # 画出可视化图形
    W = list(W.T) # 矩阵的列向量才能float
    b = float(b)  #将矩阵b转换为float b
    n = trainLabel.shape[0]
    xcord1, ycord1 = [], []
    xcord0, ycord0 = [], []
    for i in range(n):
        if int(trainLabel[i]) == 1:
            xcord1.append(trainSet[i, 0])
            ycord1.append(trainSet[i, 1])
        else:
            xcord0.append(trainSet[i, 0])
            ycord0.append(trainSet[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(xcord1, ycord1, s = 30, c = 'red')
    ax.scatter(xcord0, ycord0, s = 30, c = 'green')
    ax.set_xlim([-2.5, 12]) # 设置x坐标长度
    ax.set_ylim([-8, 6])   # 设置y坐标长度
    x = np.arange(-20, 20, 0.01)
    y = (-b - W[0] * x)/W[1]
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.show()
    


filename = r'F:\github\MachineLearning全\input\5.Logistic\HorseColicTraining.txt'
trainSet, trainLabel = loadDataSet(filename)
filename = r'F:\github\MachineLearning全\input\5.Logistic\HorseColicTest.txt'
testSet, testLabel = loadDataSet(filename)
b, alphas = smoP(trainSet, trainLabel, 0.6, 0.001, 40)
m, n = trainSet.shape
W = np.zeros((1, n))
for i in range(m):
    if float(alphas[i]) > 0:
        W += alphas[i] * trainLabel[i] * trainSet[i] # 算出W
errCount = 0
for i in range(len(testSet)):
    S = W * testSet[i].T + float(b)
    if S < 0:
        yhat = -1
    else:
        yhat = 1
    if yhat != int(testLabel[i]):
        errCount += 1
print('total error rate {}%'.format(errCount/len(testSet)))
print(alphas[alphas > 0]) #大于0 的为支撑向量SV























