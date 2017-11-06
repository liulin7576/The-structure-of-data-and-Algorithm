import numpy as np

def grabWeakClass(filename): #读取pickle的数据
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


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

def adaClassify(dataSet, weakClass):  #adaBoost分类函数
    dataSet = np.mat(dataSet)
    m = dataSet.shape[0]
    aggClass = np.mat(np.zeros((m, 1)))
    for i in range(len(weakClass)):
        classMat = stumpClassify(dataSet, weakClass[i]['dim'],\
                                 weakClass[i]['threshold'], weakClass[i]['ineq'])
        aggClass += weakClass[i]['alpha'] * classMat
    return sign(aggClass)


def test():
    filename = r'F:\github\MachineLearning全\input\5.Logistic\HorseColicTraining.txt'
    trainSet, trainLabel = loadDataSet(filename)
    m, n = trainSet.shape
    filename = r'E:\python编程处\机器学习实战\Adaboost\weakClass.txt'
    weakClass = grabWeakClass(filename)
    trainClassTestLabel = adaClassify(trainSet, weakClass)
    err = np.multiply(sign(trainClassTestLabel) != trainLabel, np.ones((m, 1)))
    errRate = float(sum(err) / m)
    print('The training error rate is {}%'.format(errRate * 100))
    filename = r'F:\github\MachineLearning全\input\5.Logistic\HorseColicTest.txt'
    testSet, testLabel = loadDataSet(filename)
    m, n = testSet.shape
    testClassTestLabel = adaClassify(testSet, weakClass)
    err = np.multiply(sign(testClassTestLabel) != testLabel, np.ones((m, 1)))
    errRate = float(sum(err) / m)
    print('The test error rate is {}%'.format(errRate * 100))

test()



    
