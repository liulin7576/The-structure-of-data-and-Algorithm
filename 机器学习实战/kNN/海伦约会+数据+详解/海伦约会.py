from numpy import *
import operator
def file2matrix(filename):  #将输入数据处理为计算机想要的数据
    fr = open(filename)
    arrayLines = fr.readlines() #读取数据所有行
    numberOfLines = len(arrayLines) #计算有多少个数据
    returnMat = zeros((numberOfLines, 3)) #保存后来的训练集
    classLabel = [] #保存标签
    index = 0
    for line in arrayLines:
        #line = line.strip()
        everyLine = line.split('\t') #按照Tab键分割
        returnMat[index, :] = everyLine[0:3]
        if everyLine[-1] == 'largeDoses\n':
            everyLine[-1] = 3
        elif everyLine[-1] == 'smallDoses\n':
            everyLine[-1] = 2
        else:
            everyLine[-1] = 1
        classLabel.append(everyLine[-1])
        index += 1
    return returnMat, classLabel #返回数据

def autoNorm(dataSet): #数据归一化，避免大树吃小数以及量纲的影响
    minVals = dataSet.min(0) #0表示按列取最小值这里为1x3的array
    maxVals = dataSet.max(0) #同上
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet)) # 存取最后的结果
    m = dataSet.shape[0] #m = 1000
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def classify(inx, dataSet, labels, k): # k近邻算法的实现
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inx, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDist = distances.argsort()
    classLabel = {}
    for i in range(k):
        voteLabel = labels[sortedDist[i]]
        classLabel[voteLabel] = classLabel.get(voteLabel, 0) + 1
    sortedClass = sorted(classLabel.items(), key = operator.itemgetter(1),
                         reverse = True)
    return sortedClass[0][0]

def datingClassTest(): #测试结果的准确性
    datingData, classLabel = file2matrix(\
        r'F:\github\Machine-Learning\kNN\2.海伦约会\datingTestSet.txt')
    testRate = 0.1 #测试数据的比例
    normMat, ranges, minVals = autoNorm(datingData)
    m = normMat.shape[0]
    numTestVecs = int(testRate * m)
    errCount = 0.0 #计数错误的数量
    for i in range(numTestVecs):
        classifyResult = classify(normMat[i, :], normMat[numTestVecs:,:],
                                  classLabel[numTestVecs:], 4)
        if  classLabel[i] != classifyResult:
            print('The class result is %d, The real answer is %d'\
              % (classifyResult, classLabel[i]))
            errCount += 1.0
    print('The total error rate is %f' %(errCount / float(numTestVecs)))
    
    
def classifyPerson(): #最终应用版本
    resultList = ['一点都不喜欢', '一般喜欢', '很喜欢']
    percentagePlay = float(input('花在游戏上的时间?'))
    miles = float(input('飞行的里程数?'))
    iceCream = float(input('吃了多少冰淇淋?'))
    datingData, classLabel = file2matrix(\
        r'F:\github\Machine-Learning\kNN\2.海伦约会\datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingData)
    test = array([miles, percentagePlay, iceCream])
    classResult = classify((test - minVals)/ranges, normMat, classLabel, 3)
    print('这个人的喜欢程度为：%s' %resultList[classResult - 1])



    

    
