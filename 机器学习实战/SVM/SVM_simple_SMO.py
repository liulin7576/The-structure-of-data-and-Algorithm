import numpy as np
import time
import random
import matplotlib.pyplot as plt
random.seed(time.time())



def loadDataSet(filename): # 读取数据,并且对数据进行处理
    f = open(filename)
    trainSet, trainLabel = [], []
    for line in f.readlines():
        lineArr = line.strip().split('\t')
        trainSet.append([float(lineArr[0]), float(lineArr[1])])
        trainLabel.append(int(lineArr[-1]))
    return np.mat(trainSet), np.mat(trainLabel).T # 全部数据和标签进行矩阵运算

def selectJrand(i, m): # 选择和i配对的j，但是和j不能是相同的
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

def clipAlpha(aj, H, L): # 限定alpha的取值范围
    if aj > H:
        aj = H
    elif aj < L:
        aj = L
    return aj

def simpleSMO(dataSet, dataLabel, C, toler, maxIter): #分别表示数据集，标签，C代表容许犯错误的大小衡量
    b = 0                                             # C越大表示允许犯错误的点越小,toler表示容错率，maxIter表示最大循环次数
    m, n = dataSet.shape
    alphas = np.mat(np.zeros((m, 1))) #langrange 因子
    iterTotal = 0
    while iterTotal < maxIter:  #只有进行了maxIter次没有循环的修改才能推出while循环
        alphaPair = 0
        for i in range(m):
            fxi = float((np.multiply(alphas, dataLabel).T* \
                                    dataSet * dataSet[i].T) + b)   #参照林轩田老师机器学习技法课程
            Ei = fxi - float(dataLabel[i])
            if ((Ei*dataLabel[i] > toler) and (alphas[i] > 0)) or \
               ((Ei*dataLabel[i] < -toler) and (alphas[i] < C)):
                j = selectJrand(0, m)
                fxj = float((np.multiply(alphas, dataLabel).T* \
                                    dataSet * dataSet[j].T) + b)
                Ej = fxj - dataLabel[j]
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if dataLabel[i] != dataLabel[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[i] + alphas[j])
                if L == H:
                    #print('L == H')
                    continue
                eta = 2.0 * dataSet[i] * dataSet[j].T - \
                      np.dot(dataSet[i], dataSet[i].T) - \
                      np.dot(dataSet[j], dataSet[j].T)
                if eta >= 0:
                    #print('eta >= 0')
                    continue
                alphas[j] -= dataLabel[j] * (Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if(abs(alphas[j] - alphaJold) < 0.00001):
                    #print('J not moving enough!')
                    continue
                alphas[i] += dataLabel[j] * dataLabel[i] *\
                             (alphaJold - alphas[j])
                b1 = b - Ei - dataLabel[i] * (alphas[i] - alphaIold) *\
                     dataSet[i, :] * dataSet[i, :].T - \
                     dataLabel[j] * (alphas[j] - alphaJold) *\
                     dataSet[i, :] * dataSet[j, :].T
                b2 = b - Ej - dataLabel[i] * (alphas[i] - alphaIold) *\
                     dataSet[i, :] * dataSet[j, :].T - \
                     dataLabel[j] * (alphas[j] - alphaJold) *\
                     dataSet[j] * dataSet[j].T
                if (alphas[i] > 0) and (alphas[i] < C):
                    b = b1
                elif (alphas[j] > 0) and (alphas[j] < C):
                    b = b2
                else:
                    b = (b1 + b2)/2
                alphaPair += 1
        if alphaPair == 0:
            iterTotal += 1
        else:
            iterTotal =0
        print('iteration number %d' %iterTotal)
    return alphas, b

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
    


filename = r'F:\github\MachineLearning全\input\6.SVM\testSet.txt'
trainSet, trainLabel = loadDataSet(filename)
alphas, b = simpleSMO(trainSet, trainLabel, 0.6, 0.001, 40)
m, n = trainSet.shape
W = np.zeros((1, n))
for i in range(m):
    if float(alphas[i]) > 0:
        W += alphas[i] * trainLabel[i] * trainSet[i] # 算出W
plotBestBound(trainSet, trainLabel, W, b)
print(alphas[alphas > 0]) #大于0 的为支撑向量SV








