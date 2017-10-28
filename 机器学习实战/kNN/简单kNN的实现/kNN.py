from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    label = ['A','A','B','B']
    return group, label

def classify(inx, dataSet, label, k):
    dataSetSize = dataSet.shape[0] #算出行数
    diffMat = tile(inx, [dataSetSize, 1]) - dataSet #对应的相减
    sqDiffMat = diffMat ** 2 #各元素平方
    sqDistances = sum(sqDiffMat, axis = 1) #行元素相加
    Distances = sqDistances ** 0.5 #得到距离值
    sortedDict = Distances.argsort() #返回距离值得从小到大排列的下标
    classSort = {}
    for i in range(k):
        voteLabel = label[sortedDict[i]]
        classSort[voteLabel] = classSort.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classSort.items(), key = operator.itemgetter(1),
                              reverse = True)
    return sortedClassCount[0][0] #返回分类结果
    
