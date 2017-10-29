from math import log
import operator

def createData():
    f=open(r'F:\github\MachineLearning全\input\3.DecisionTree\lenses.txt')
    dataSet = [example.strip().split('\t') for example in f.readlines()]# strip()可以把'\n'去掉
    label = ['age', 'prescript', 'astigmatic', 'tearRate']
    f.close()
    return dataSet, label
    
def calcShannon(dataSet): #计算香农熵
    numEntries = len(dataSet)
    labelCounts = {}
    for i in dataSet:
        currentLabel = i[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = labelCounts[key] / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt
        
def splitData(dataSet, axis, value):#axis是划分数据集的特征，value是需要返回的特征值
    retDataSet = []
    for vec in dataSet:
        if vec[axis] == value:
            midVec = vec[ :axis]
            midVec.extend(vec[axis+1:])
            retDataSet.append(midVec)
    return retDataSet

def chooseBestFeature(datsSet): # 选择最好的特征
    numFeatures = len(datsSet[0]) - 1  #特征的数目
    bestInfoGain, bestFeature = 0, -1  #初始化选择的特征和信息增益
    baseEntropy = calcShannon(dataSet) # 计算初始数据的香农熵
    for i in range(numFeatures):
        featureList = [example[i] for example in dataSet] #挑选出所有第一列的特征
        uniqueVal = set(featureList) #比如第一列结果为[1,1,1,0,0]，那么只有两个有效的特征
        newEntropy = 0.0
        for value in uniqueVal:  #下面的就比较简单了
            subData = splitData(dataSet, i, value)
            prob = len(subData) / len(dataSet)
            newEntropy += prob * calcShannon(subData)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
            
def majorityCnt(classList): #叶子节点不唯一时，少数服从多数
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1),\
                              reverse = True)
    return sortedClassCount[0][0] #返回最大的类别数目作为叶子节点的类型

def createTree(dataSet, label): #创建决策树，这里的label是具体的特征，而非0,1
    classList = [example[-1] for example in dataSet] #提取所有的分类情况
    if  classList.count(classList[0]) == len(classList):#分类特征用完以后且全部分类成功
        return classList[0]
    if len(dataSet[0]) == 1: #分类特征用完但是没有分类完全
        return majorityCnt(classList)
    bestFea = chooseBestFeature(dataSet) #选择最好的特征
    bestFeaLabel = label[bestFea] #对应的最好特征的名字
    myTree = {bestFeaLabel:{}} #创建决策树，用字典表示
    del(label[bestFea]) #特征已经使用，那么删除该特征
    featValue = [example[bestFea] for example in dataSet] #选择最好特征有多少个分类情况
    uniqueVal = set(featValue) #去除重复的特征分类情况
    for value in uniqueVal:
        subLabel = label[:]
        myTree[bestFeaLabel][value] = createTree(splitData(dataSet, bestFea,\
                                                value), subLabel)
    return myTree
    
def classify(inputTree, featLabels, testVec): # 测试数据
    firstStr = list(inputTree.keys())[0]
    featIndex = featLabels.index(firstStr)
    secondDict = inputTree[firstStr]
    for key in secondDict:
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename): #将树序列化存储
    import pickle
    fw = open(filename, 'wb') # pickle要用二进制存入和二进制读取
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename): #将树取出来
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


dataSet, label = createData()
myTree = createTree(dataSet, label)
print(myTree)
dataSet, label = createData()
filename = r'E:\python编程处\机器学习实战\决策树\storeTree.txt'
storeTree(myTree, filename)
