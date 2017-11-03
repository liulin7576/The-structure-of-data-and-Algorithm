import numpy as np
import random
import re
import time
import feedparser
import operator
random.seed(time.time())


def textParse(bigString): # 处理输入文本
    splitString = re.split(r'\w*', bigString)
    return [tok.lower() for tok in splitString if len(tok) > 2]

def createVocab(text): # 构建一个所有词的集合
    vocabList = set([])
    for i in text:
        vocabList = vocabList | set(i)
    return list(vocabList)

def bagOfWords2Vec(vocabList, inputList): #将输入的把文本转换为向量形式[1,0...,1]
    words2Vec = [0] * len(vocabList)
    for i in inputList:
        if i in vocabList:
            words2Vec[vocabList.index(i)] += 1
    return words2Vec

def trainNB0(trainMat, classLabel): # 利用训练样本计算各词出现的概率
    numberOfTexts = len(trainMat)
    pA0, pA1 = np.ones(len(trainMat[0])), np.ones(len(trainMat[0]))
    p0Dem, p1Dem = 2, 2
    pAbusive = sum(classLabel)/numberOfTexts
    for i in range(numberOfTexts):
        if classLabel[i] == 1:
            pA1 += trainMat[i]
            p1Dem += sum(trainMat[i])
        else:
            pA0 += trainMat[i]
            p0Dem += sum(trainMat[i])
    p1Vect = np.log(pA1/p1Dem)
    p0Vect = np.log(pA0/p0Dem)
    return p1Vect, p0Vect, pAbusive

def classify(vec, p1Vect, p0Vect, pAbusive): # 计算某个文本属于1 or 0 ？
    p1 = np.dot(vec, p1Vect) +  np.log(pAbusive)
    p0 = np.dot(vec, p0Vect) +  np.log(1 - pAbusive)
    if p1 > p0:
        return 1
    else:
        return 0

def calcMostFre(vocabList, fullText): #计算出现频率最高的前30个词
    freqDict = {}
    for i in vocabList:
        freqDict[i] = fullText.count(i)
    sortedFreq = sorted(freqDict, key = operator.itemgetter(1),\
                        reverse = True)
    return sortedFreq[:30]

def localWords(feed1, feed0): # 测试
    docList, fullText, classList = [], [], []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    print(minLen)
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        fullText.extend(wordList)
        docList.append(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocab(docList)
    top30Words = calcMostFre(vocabList, fullText)
    for pairw in top30Words:
        if pairw[0] in vocabList:
            vocabList.remove(pairw[0])
    trainMat, classLabel = [], []
    trainSet, testSet = [i for i in range(2 * minLen)], []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randIndex])
        del(trainSet[randIndex])
    for docIndex in trainSet:
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        classLabel.append(classList[docIndex])
    p1Vect, p0Vect, pAbusive = trainNB0(trainMat, classLabel)
    errCount = 0.0
    for i in testSet:
        words2Vec =  bagOfWords2Vec(vocabList, docList[i])
        testLabel = classify(words2Vec, p1Vect, p0Vect, pAbusive)
        if testLabel !=classList[i]:
            errCount += 1
    print('total error rate is {} %'.format(errCount/len(testSet)*100)) # 打出60%的格式形式
    return p1Vect, p0Vect, vocabList

def getTopWords(ny, sf): #输出最有表征性的词语
   p1Vect, p0Vect, vocabList = localWords(ny, sf)
   topNY, topSF = [], []
   for i in range(len(p1Vect)):
       if p1Vect[i] > -6:
           topNY.append((vocabList[i], p1Vect[i]))
       if p0Vect[i] > -6:
           topSF.append((vocabList[i], p0Vect[i]))
   sortedSF = sorted(topSF, key = lambda pair:pair[1],reverse = True)
   sortedNY = sorted(topNY, key = lambda pair:pair[1],reverse = True)
   print('***********NY**********')
   for item in sortedNY:
       print(item[0])
   print('***********SF**********')
   for item in sortedSF:
       print(item[0])
        
    




# 结果检测
feed1 = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
feed0 = feedparser.parse('http://oxford.craigslist.org/stp/index.rss')
getTopWords(feed1, feed0)











