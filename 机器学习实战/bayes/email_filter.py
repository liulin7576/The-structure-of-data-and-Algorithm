#电子邮件过滤

import numpy as np
import re
import random
import time #产生随机种子
random.seed(time.time())

def vocabCreate(docList): # 构建词带集合
    vocabList = set([])
    for i in docList:
        vocabList = vocabList | set(i)
    return list(vocabList)
    

def bagOfWords2Vec(vocabList, inputSet): # 创建bayes词袋
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print('the word %s is not in my vocabulary!' %word)
    return returnVec


def trainNB0(trainMatrix, trainCategory): #trainCategory是上面的[0, 1, 0, 1, 0, 1]     numTrainDocs = len(trainMatrix)
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])   #trainMatrix是一堆[1,0,...1]的矩阵
    pAbusive = sum(trainCategory) / float(numTrainDocs) #侮辱性文档占的比重
    p1Num, p0Num = np.ones(numWords), np.ones(numWords)
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
    p1Vect = np.log(p1Vect) #用np.log()的原因是为了防止概率太小连乘导致为0
    p0Vect = np.log(p0Vect)
    return p1Vect, p0Vect, pAbusive



def textSplit(bigString):  # 分割文本
    listOfTokens = re.split(r'\w*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def classify(vec2Classify, p0Vect, p1Vect, pAbusive): # 最终分类判别函数
    p1 = np.sum(vec2Classify * p1Vect) + np.log(pAbusive)
    p0 = np.sum(vec2Classify * p0Vect) + np.log(1 - pAbusive)
    if p1 > p0:
        return 1
    else:
        return 0


def spamTest():  # 分类文本
    docList = []; classList = []; fullText = []
    for i in range(1, 26):#共有26个文本
        wordList = textSplit(open(r'F:\github\MachineLearning全\input\4.NaiveBayes\email\spam\%d.txt' %i).read())
        docList.append(wordList)
        classList.append(1) # 添加1类标签，即侮辱性文本
        fullText.extend(wordList)
        fr = open(r'F:\github\MachineLearning全\input\4.NaiveBayes\email\ham\%d.txt' %i)
        wordList = textSplit(fr.read())
        docList.append(wordList)
        classList.append(0) #添加0类标签
        fullText.extend(wordList)
    totalList = vocabCreate(docList) # 构建所有词集合
    errTotal = 0.0
    for j in range(10): # 10次交叉验证
        trainingSet = [i for i in range(50)]; testSet = [] # 不能用range产生index,因为没法用del
        for i in range(10): # 随机产生10个测试文件
            randIndex = int(random.uniform(0, len(trainingSet)))
            testSet.append(trainingSet[randIndex])
            del(trainingSet[randIndex]) # 删除已经随机使用的index
        trainMat = []; trainClass = []
        for docIndex in trainingSet:
            trainMat.append(bagOfWords2Vec(totalList, docList[docIndex]))
            trainClass.append(classList[docIndex])
        p1Vect, p0Vect, pAbusive = trainNB0(trainMat, trainClass)
        errCount = 0.0
        for docIndex in testSet:
            wordVector = bagOfWords2Vec(totalList, docList[docIndex])
            testLabel = classify(wordVector, p0Vect, p1Vect, pAbusive) 
            if testLabel != classList[docIndex]:
                print('文件%d分错啦' %i)
                errCount += 1
        errTotal += errCount
    print('arter 10 validations, the total err is %f' %(errTotal/(10 * len(testSet))))



        
    
