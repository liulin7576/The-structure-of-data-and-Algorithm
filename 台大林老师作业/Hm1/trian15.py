import numpy as np

def dataSelect(filename): # 分离数据
    f = open(filename)
    fm = f.readlines()
    trainSet = np.zeros((len(fm), 5))
    classLabel = []
    index = 0
    for line in fm:
        line.strip()
        splitLine = line.split('\t')
        trainSet[index, 0] = 1
        trainSet[index, 1:] = splitLine[ :4]
        label = splitLine[-1].split('\n')
        classLabel.append(int(label[0]))
        index += 1
    return trainSet, classLabel

def PLA(trainSet, classLabel):
    w = np.zeros((1, 5))
    countAll = 0
    for j in range(2000):
        countIter = 0
        arr = np.arange(len(trainSet))
        np.random.shuffle(arr)
        for i in arr:
            buff = np.dot(w, trainSet[i].T)
            if buff <= 0:
                
                yhat = -1
            else:
                yhat = 1
            if yhat != classLabel[i]:
                w = w + 0.5 * classLabel[i] * trainSet[i]
                break
            countIter +=1
        countAll += 1
        if countIter == len(trainSet):
            return countAll
    
trainSet, classLabel = dataSelect(r'F:\机器学习\台大林老师作业\Hm1\train_15.txt')
countAll = PLA(trainSet, classLabel)
count = 0
for i in range(2000):
    count += countAll
print(count / 2000)
