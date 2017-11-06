import numpy as np


def loadSimpleSet():  #初始化一个简单的数据集
    dataSet = np.mat([[1, 2.1], [2, 1.1],
                     [1.3, 1], [1, 1], [2, 1]])
    dataLabel = [1, 1, -1, -1, 1]
    return np.mat(dataSet), np.mat(dataLabel).T

def loadDataSet(filename): # 加载第五章数据集
    f = open(filename)
    dataMat, dataLabel = [], []
    for line in f.readlines():
        lineArr = line.strip().split('\t')
        buff = list(map(float, lineArr))
        dataMat.append(buff[:-1])
        if buff[-1] == 0:
            dataLabel.append(-1)
        else:
            dataLabel.append(1)
    return np.mat(dataMat), np.mat(dataLabel).T


def sign(x): #定义一个符号函数
    m = x.shape[0]
    returnMat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        if x[i] > 0:
            returnMat[i] = 1
        else:
            returnMat[i] = -1
    return returnMat

def stumpClassify(dataMat, dimen, threshold, threshIneq): #给定数据矩阵， 第dimen个特征，阈值，左边还是右边为+1的判断
    returnMat = np.mat(np.ones((len(dataMat), 1)))
    if threshIneq == 'lt':
        returnMat[dataMat[:, dimen] < threshold] = -1
    else:
        returnMat[dataMat[:, dimen] > threshold] = -1
    return returnMat

def buildStump(dataSet, dataLabel, D): #创建单层决策树
    m, n = dataSet.shape
    numStep, bestStump, bestClassEst = 10, {}, np.mat(np.zeros((m, 1)))
    minErr = float('inf')
    for i in range(n):  #对每一个特征进行遍历
        rangeMin, rangeMax = dataSet[:, i].min(), dataSet[:, i].max()
        stepSize = (rangeMax - rangeMin)/numStep
        for j in range(-1, numStep + 1):
            for inequal in ['lt', 'gt']: #判断小于取正号还是大于threshold取正号
                threshold = rangeMin + j * stepSize
                predict = stumpClassify(dataSet, i, threshold, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predict == dataLabel] =  0
                weightError = D.T * errArr
                #print('split dim %d, threshold %.2f, the weightError %.2f'%(i, threshold, weightError))
                if weightError < minErr:
                    minErr = weightError
                    bestClassEst = predict.copy()
                    bestStump['dim'] = i
                    bestStump['threshold'] = threshold
                    bestStump['ineq'] = inequal
    return bestStump, minErr, bestClassEst

def adaBoostDTree(dataSet, dataLabel, numIter = 40): #构建adaBoost决策树
    m, n = dataSet.shape
    weakClass = []
    D = np.mat(np.ones((m, 1)))/m
    aggClass = np.mat(np.zeros((m, 1))) #加权综合后的类别情况记录
    for i in range(numIter):
        bestStump, minErr, bestClassEst = buildStump(dataSet, dataLabel, D)
        alpha = float(0.5 * np.log((1 - minErr)/max(minErr, 1e-16))) #防止除数为0用max函数
        bestStump['alpha'] = alpha
        weakClass.append(bestStump)
        expon = np.multiply(-alpha * dataLabel, bestClassEst)
        D = np.multiply(D, np.exp(expon))
        D = D/sum(D)
        aggClass += alpha * bestClassEst
        err = np.multiply(sign(aggClass) != dataLabel, np.mat(np.ones((m, 1))))
        errRate = float(sum(err) / m)
        print('total errRate is {}%'.format(errRate * 100) )
        if errRate == 0:
            break
    return weakClass, aggClass


def adaClassify(dataSet, weakClass):  #adaBoost分类函数
    dataSet = np.mat(dataSet)
    m = dataSet.shape[0]
    aggClass = np.mat(np.zeros((m, 1)))
    for i in range(len(weakClass)):
        classMat = stumpClassify(dataSet, weakClass[i]['dim'],\
                                 weakClass[i]['threshold'], weakClass[i]['ineq'])
        aggClass += weakClass[i]['alpha'] * classMat
    return sign(aggClass)

def storeAda(weakClass, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(weakClass, fw)
    fw.close()

def plotROC(predStrength, classLabel):  #画出ROC去曲线
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClass = np.sum(classLabel == 1)
    yStep = 1/float(numPosClass)
    xStep = 1/float(classLabel.shape[0] - numPosClass)
    sortedInd = predStrength.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(1, 1, 1)
    for index in sortedInd:
        if classLabel[index] == 1:
            delX, delY = 0, yStep
        else:
            delX, delY = xStep, 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c = 'b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('FP')
    plt.ylabel('TP')
    plt.title('ROC curve of horse')
    plt.show()
    print('The Area under the Cure is', ySum * xStep)


filename = r'F:\github\MachineLearning全\input\5.Logistic\HorseColicTraining.txt'
trainSet, trainLabel = loadDataSet(filename)
weakClass, aggClass = adaBoostDTree(trainSet, trainLabel, numIter = 50)
filename = r'E:\python编程处\机器学习实战\Adaboost\weakClass.txt'
storeAda(weakClass, filename)
pred = []
for i in aggClass:
    pred.append(float(i))
plotROC(np.array(pred),trainLabel)


'''        
dataSet, dataLabel = loadSimpleSet()
weakClass = adaBoostDTree(dataSet, dataLabel)
classResult = adaClassify([[5 ,5], [0, 0]], weakClass)
print(classResult)
'''





   
    
    
