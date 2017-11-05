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
        trainLabel.append(int(float(lineArr[-1])))
    for i in range(len(trainLabel)):
        if trainLabel[i] == 0:
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


def kernelTrans(X, A, kTup): #kTup是表明要使用的核函数类型
    m, n = X.shape
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for i in range(m):
            deltaRow = X[i, :] - A
            K[i, :] = deltaRow * deltaRow.T
        K = np.exp(-K/(kTup[1]**2))
    else:
        raise NameEroor('No kernel like this!')
    return K
        

    

class optStruct:  #构造类存储所需的变量，便于访问
    def __init__(self, dataSet, dataLabel, C, toler, kTup):
        self.dataSet = dataSet
        self.dataLabel = dataLabel
        self.C = C
        self.toler = toler
        self.m = len(dataSet)
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.cache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.dataSet, self.dataSet[i, :], kTup)


def calcEk(oS, k): # 计算第k个样本和预测之间的误差
    fxk = np.multiply(oS.alphas, oS.dataLabel).T * \
          oS.K[:, k] + oS.b
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
        eta = 2 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
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
            oS.K[i, i] -oS.dataLabel[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.dataLabel[i] * (oS.alphas[i] - alphaIold) * \
             oS.K[i, j] -oS.dataLabel[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
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
    oS = optStruct(dataSet, dataLabel, C, toler, kTup)
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

filename = r'F:\github\MachineLearning全\input\5.Logistic\HorseColicTraining.txt'
trainSet, trainLabel = loadDataSet(filename)
filename = r'F:\github\MachineLearning全\input\5.Logistic\HorseColicTest.txt'
testSet, testLabel = loadDataSet(filename)
b, alphas = smoP(trainSet, trainLabel, 0.6, 0.001, 40)
svInd = np.nonzero(alphas.A > 0)[0]
svs = trainSet[svInd]
labelSV = trainLabel[svInd]
print('There are %d Support vector' %len(svs))
m, n = testSet.shape
W = np.zeros((1, n))
errCount = 0
for i in range(m):
    kernelEval = kernelTrans(svs, testSet[i, :], ('rbf', 1.3))
    yhat = kernelEval.T * np.multiply(labelSV, alphas[svInd])
    if yhat < 0:
        yhat = -1
    else:
        yhat = 1
    if yhat != int(testLabel[i]):
        errCount += 1
print('total error rate {}%'.format(errCount/len(testSet) * 100))























