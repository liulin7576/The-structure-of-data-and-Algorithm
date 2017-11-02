import numpy as np
import random
import time
random.seed(time.time()) # 产生随机种子

def dataSelect(filename): # 分离数据
    f = open(filename)
    fm = f.readlines()
    trainSet = np.zeros((len(fm), 5))
    classLabel = []
    index = 0
    for line in fm:
        line.strip()
        splitLine = line.split('\t')
        trainSet[index, 0] = 1               # 加x0 = 1
        trainSet[index, 1:] = splitLine[ :4] # 添加其他的输入特征
        label = splitLine[-1].split('\n')
        classLabel.append(int(label[0]))
        index += 1
    return trainSet, classLabel

def Pocket(trainSet, classLabel): #pocket算法
    w = np.zeros((1, 5))
    totalCount = 0
    while True: # 在满足条件之前都进行更新迭代
         buff_j = int(random.uniform(0, len(trainSet))) # 随机产生一个输入样本标号
         buff = np.dot(w, trainSet[buff_j].T)
         if buff > 0:
             yhat =1
         else:
             yhat =-1
         if yhat != classLabel[buff_j]:
             w = w + classLabel[buff_j] * trainSet[buff_j, :]
             totalCount += 1
         if totalCount == 100: # 100次更新之后停止迭代
             break
    return w

def testResult(w, testSet, testLabel): #验证pocket算法
    totalErr = 0
    for j in range(len(testSet)):
        buff = np.dot(w, testSet[j].T)
        if buff > 0:
             yhat =1
        else:
             yhat =-1
        if yhat != testLabel[j]:
             totalErr += 1
    return totalErr / len(testLabel)

    
    
trainSet, classLabel = dataSelect(r'F:\机器学习\台大林老师作业\Hm1\train_18.txt')
testSet, testLabel = dataSelect(r'F:\机器学习\台大林老师作业\Hm1\test_18.txt')
totalErrRate = 0
for i in range(2000):  # 进行2000次迭代
    w = Pocket(trainSet, classLabel)
    totalErrRate += testResult(w, testSet, testLabel)
print('total error rate:%f' % (totalErrRate / 2000))
