from numpy import *
from math import log

def loadDataSet(): #手动创建一个单词集合
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1] #1代表侮辱词汇
    return postingList, classVec

def createVocab(dataSet):
    vocabList = set([]) # 初始化为集合，这样便于后续用集合的并集操作
    for document in dataSet:
        vocabList = vocabList | set(document)
    return list(vocabList)

def words2vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word %s is not in my vocabulary!' %word)
    return returnVec

def trainNB0(trainMatrix, trainCategory): #trainCategory是上面的[0, 1, 0, 1, 0, 1]     numTrainDocs = len(trainMatrix)
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])   #trainMatrix是一堆[1,0,...1]的矩阵
    pAbusive = sum(trainCategory) / float(numTrainDocs) #侮辱性文档占的比重
    p1Num, p0Num = ones(numWords), ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i] #array数组对应元素相加
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom #array数组对应元素相乘
    p0Vect = p0Num / p0Denom
    for i in range(len(p1Vect)):
        p1Vect[i] = log(p1Vect[i]) #用Log的原因是为了防止概率太小连乘导致为0
        p0Vect[i] = log(p0Vect[i])
    return p1Vect, p0Vect, pAbusive

def classify(vec2Classify, p0Vect, p1Vect, pAbusive): #分类函数
    p1 = sum(vec2Classify * p1Vect) + log(pAbusive) #计算文档属于1的概率
    p0 = sum(vec2Classify * p0Vect) + log(1 - pAbusive)
    if p1 > p0:
        return 1
    else:
        return 0


def testing(): #测试函数
    dataSet, classVec = loadDataSet()
    vocabList = createVocab(dataSet)
    trainMatrix = []
    for i in dataSet:
        trainMatrix.append(words2vec(vocabList, i))
    p1Vect, p0Vect, pAbusive = trainNB0(trainMatrix, classVec)#p1Vect是numpy类型
    test1 = ['cute', 'love', 'help']
    test1Vec = words2vec(vocabList, test1) #test1Vec不是numpy类型
    print(test1,'的分类结果为:'+ str(classify(test1Vec, p0Vect, p1Vect, pAbusive)))
    test2 = ['stupid', 'garbage']
    test2Vec = words2vec(vocabList, test2)
    print(test2,'的分类结果为:'+ str(classify(test2Vec, p0Vect, p1Vect, pAbusive)))





testing()    #测试一下
'''
    >>> b = array([1, 2, 3])
    >>> a = [1, 2, 3]
    >>> a * b 结果为：array([1, 4, 9])
'''         
