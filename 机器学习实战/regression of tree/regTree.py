'''
要熟练使用tolist方法将矩阵转换为列表，很实用！如：dataSet[:, 1].T.tolist()[0]
'''
import numpy as np
import matplotlib.pyplot as plt




def loadData(filename):  #加载数据
    f = open(filename)
    dataMat = []
    for line in f.readlines():
        lineArr = line.strip().split('\t')
        buff = list(map(float, lineArr))
        dataMat.append(buff)
    return np.mat(dataMat)  #一起存储的原因不会影响后续结果， 因为后面的操作都是衡量最后一列即标签的混乱度，所以可以放在一起存储而不用分开


def binSplitData(dataSet, feature, value):  #按照特征值value分割特征feature为左子树和右子树
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value), :][0]  #这里的[0]表示的是取第一个值，因为前面的是一个包含三个括号的matrix
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value), :][0]
    return mat0, mat1

def createTree(dataSet,leafType, errType,  ops = (1,4)): #创建回归树, 这里的ops第一个数字表示误差的变化范围，如果小于这个值表示可以直接当成一个叶子节点处理，
                                        #第二个数字表示如果分割后的数据大小小于这个值，那么也不对这个特征值做分割
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitData(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


'''
下面的两个函数是用来产生只有一个值的叶节点
'''
def regLeaf(dataSet):   #计算最后一列的数据平均值，也就是分为叶子节点的时候直接取的平均值
    return np.mean(dataSet[:, -1])


def regErr(dataSet):  #计算最后一列的混乱度，这里用总方差来衡量
    return dataSet.shape[0] * np.var(dataSet[:, -1])


'''
下面的三个函数是用来产生有线段模型的叶节点
'''
def linearSolve(dataSet):  #建立dataSet的线性模型
    m, n = dataSet.shape
    X, Y = np.mat(np.ones((m, n))), np.mat(np.ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]
    XTX = X.T * X
    if np.linalg.det(XTX) == 0:
        print('No inverse!')
        return
    ws = XTX.I * X.T * Y
    return ws, X, Y

def modelLeaf(dataSet):   #类似上面的regLeaf函数
    ws, X, Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet): #类似上面的regErr函数
    ws, X, Y = linearSolve(dataSet)
    yhat = X * ws
    return np.sum(np.square(yhat - Y))  #真的是无语，这边写成np.square(yhat, Y)，然后找了30分钟的错误，才找出来。我服了！




def chooseBestSplit(dataSet, leafType, errType, ops = (1, 4)):  #选择最好的二元切分方式
    tolS, tolN = ops[0], ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1: #这里的tolist()是将矩阵转换为列表
        return None, leafType(dataSet)
    m, n = dataSet.shape
    S = errType(dataSet)
    bestS, bestIndex, bestVal = np.inf, 0, 0
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            mat0, mat1 = binSplitData(dataSet, featIndex, splitVal)
            if (mat0.shape[0] < tolN) or (mat1.shape[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestS = newS
                bestIndex = featIndex
                bestVal = splitVal
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    return bestIndex, bestVal

def plotScatter(dataSet):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(dataSet[:, -2].T.tolist()[0], dataSet[:, -1].T.tolist()[0], c = 'red')
    plt.show()


def isTree(obj):  #用来判断树是否为字典，也就是是不是叶子节点
    return (type(obj).__name__ == 'dict')


def getMean(tree):   #将左子树和右子树合并
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['right'] + tree['left'])/2

def prune(tree, testData):  #执行后剪枝函数
    if testData.shape[0] == 0:  #处理没有测试数据的情况
        return getMean(tree)
    if (isTree(tree['left']) or isTree(tree['right'])):
        lSet, rSet = binSplitData(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if (not isTree(tree['left'])) and (not isTree(tree['right'])):
        lSet, rSet = binSplitData(testData, tree['spInd'], tree['spVal'])
        errNoMerge = np.sum(np.square(tree['left'] - lSet[:, -1])) + \
                     np.sum(np.square(tree['right'] - rSet[:, -1]))
        treeMean = (tree['left'] + tree['right'])/2
        errMerge = np.sum(np.square(testData[:, -1] - treeMean))
        if errNoMerge > errMerge:
            print('merge')
            return treeMean
        else:
            return tree
    else:
        return tree

def test(tree, testData):  #测试一个样例，树模型预测
    while isTree(tree):
        lSet, rSet = binSplitData(testData, tree['spInd'], tree['spVal'])
        if len(lSet) == 0:
            tree = tree['right']
        else:
            tree = tree['left']
    return tree[0] + testData * tree[1]
        
def regTest(tree, testData): #树回归预测
    while isTree(tree):
        lSet, rSet = binSplitData(testData, tree['spInd'], tree['spVal'])
        if len(lSet) == 0:
            tree = tree['right']
        else:
            tree = tree['left']
    return tree
        
    

def validTest():
    filename = r'F:\github\MachineLearning全\input\9.RegTrees\bikeSpeedVsIq_train.txt'
    dataSet = loadData(filename)
    #plotScatter(dataSet)
    retTree = createTree(dataSet, modelLeaf, modelErr, ops = (1, 20))
    print(retTree)
    filename = r'F:\github\MachineLearning全\input\9.RegTrees\bikeSpeedVsIq_test.txt'
    dataSet = loadData(filename)
    m,n = dataSet.shape
    yhat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yhat[i] = test(retTree, dataSet[i,:-1])
    print('R2值为：',np.corrcoef(yhat, dataSet[:, -1], rowvar = 0)[0, 1])


def apiModelTest(tree, dataSet):  #用于api的数据预测返回结果,树模型
    m,n = dataSet.shape
    yhat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yhat[i] = test(tree, dataSet[i,:])
    return yhat

def apiRegTest(tree, dataSet):  #用于api的数据预测返回结果，树回归
    m,n = dataSet.shape
    yhat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yhat[i] = regTest(tree, dataSet[i,:])
    return yhat


if __name__ == '__main__':
    validTest()




















