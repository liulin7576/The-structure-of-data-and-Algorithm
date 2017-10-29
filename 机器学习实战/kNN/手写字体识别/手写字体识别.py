from numpy import *
import operator
from os import listdir

def img2vector(filename): #将输入图片格式转换为1x1024的数组
    returnVect = zeros((1, 1024)) #创建一个array数组
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            returnVect[0, 32*i + j] = int(line[j])
    return returnVect

def classify(inx, dataSet, label, k): # k近邻分类器(具体解释见前面)
    m = dataSet.shape[0]
    sqMat = tile(inx, (m, 1)) - dataSet
    sq2Mat = sqMat ** 2
    sqDistances = sq2Mat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDist = distances.argsort()
    classLabel = {}
    for i in range(k):
        voteLabel = label[sortedDist[i]]
        classLabel[voteLabel] = classLabel.get(voteLabel, 0) + 1
    classLabelResult = sorted(classLabel.items(), key = operator.itemgetter(1),\
                              reverse = True)
    return classLabelResult[0][0]

def handwritingClass():
    hwLabels = [] # 存储训练样本的标签
    trainList = listdir(r'F:\github\MachineLearning全\input\2.KNN\trainingDigits')
    m = len(trainList)
    trainMat = zeros((m, 1024))
    for i in range(m):
        fileName = trainList[i]
        fileStr = fileName.split('.')[0]
        classLabel = int(fileStr.split('_')[0]) #得到了类别，该名称第一位数就是类别
        hwLabels.append(classLabel)
        trainMat[i, :] = img2vector(r'F:\github\MachineLearning全\input\2.KNN\trainingDigits\%s'\
                                    %fileName)
    testList = listdir(r'F:\github\MachineLearning全\input\2.KNN\testDigits')
    n = len(testList)
    err = 0.0
    for i in range(n):
        fileName = testList[i]
        fileStr = fileName.split('.')[0]
        trueLabel = int(fileStr.split('_')[0])
        testVector = img2vector(r'F:\github\MachineLearning全\input\2.KNN\testDigits\%s'\
                                    %fileName)
        testResult = classify(testVector, trainMat, hwLabels, 3)
        if testResult != trueLabel:
            err +=1.0
            print("original label is %d, but now is %d" %(trueLabel, testResult))
    print('total error rate is %f' %(err / float(n))) #打印错误率 1.057%
