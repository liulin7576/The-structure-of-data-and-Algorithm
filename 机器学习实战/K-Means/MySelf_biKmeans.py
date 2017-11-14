import numpy as np
import matplotlib.pyplot as plt

def loadData(filename):   #读入数据并做处理
    f = open(filename)
    dataSet = []
    for line in f.readlines():
        lineArr = line.strip().split('\t')
        lineArr = list(map(float, lineArr))
        dataSet.append(lineArr)
    return np.mat(dataSet)


def computeDis(vec1, vec2):   #计算两个点距离
    return np.sqrt(np.sum(np.square(vec1 - vec2)))

def randCent(dataSet, k):  #随机选择k个簇中心
    n = dataSet.shape[1]
    centers = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = np.min(dataSet[:, j], axis = 0)
        rangeJ = float(np.max(dataSet[:, j], axis = 0) - minJ)
        centers[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centers

def plotDot(dataSet, centers):  #用图形化展示出来
    fig = plt.figure()
    ax= fig.add_subplot(1, 1, 1)
    ax.scatter(dataSet[:, 0].T.tolist()[0], dataSet[:, 1].T.tolist()[0], c = 'green')
    ax.scatter(centers[:, 0].T.tolist()[0], centers[:, 1].T.tolist()[0], c = 'red')
    plt.show()

def plotDiverse(dataSet, k, bestCenters, cluster):  #和上面的画图进行对比
    scatterMarkers = ['s', 'o', '^', '8', 'p']
    bestCenters = np.mat(bestCenters)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(k):
        currentDataSet = dataSet[np.nonzero(cluster[:, 0] == i)[0], :]
        markerStyle = scatterMarkers[i]
        ax.scatter(currentDataSet[:, 0].T.tolist()[0], currentDataSet[:, 1].T.tolist()[0],\
                   marker = markerStyle, s = 90)
    ax.scatter(bestCenters[:, 0].T.tolist()[0], bestCenters[:, 1].T.tolist()[0],\
               marker = '+', s = 300)
    plt.show()


def kMeans(dataSet, k, createCent = randCent, distMeans = computeDis):  #KMeans的聚类实现
    m, n = dataSet.shape
    centers = createCent(dataSet, k)
    cluster = np.mat(np.zeros((m, 2)))
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist, minIndex = np.inf, -1
            for j in range(k):
                distJI = distMeans(dataSet[i, :], centers[j, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if cluster[i, 0] != minIndex:
                cluster[i, 0] = minIndex
                cluster[i, 1] = minDist ** 2
                clusterChanged = True
        for cent in range(k):
            buff = dataSet[np.nonzero(cluster[:, 0] == cent)[0]]
            centers[cent, :] = np.mean(buff, axis = 0)
    return centers, cluster
            

def biKmeans(dataSet, k, distMeans = computeDis):  #二分聚类,防止收敛到局部最小值
    m, n = dataSet.shape
    cluster = np.mat(np.zeros((m, 2)))
    center = np.mean(dataSet, axis = 0).tolist()[0]
    centerList = [center]
    for j in range(m):
        cluster[j, 1] = distMeans(np.mat(center), dataSet[j, :]) ** 2
    while len(centerList) < k:
        lowestSSE, bestClassify = -np.inf, np.inf
        for i in range(len(centerList)):
            currentCluster = dataSet[np.nonzero(cluster[:, 0] == i)[0], :]
            centerBuff, clusterBuff = kMeans(currentCluster, 2)
            sseSplit = np.sum(clusterBuff[:, 1])
            sseNotSplit = np.sum(cluster[np.nonzero(cluster[:, 1] == i)[0], 1])
            if sseNotSplit > lowestSSE  and (sseNotSplit - sseSplit) < bestClassify:
                lowest = sseNotSplit
                bestSplit = i
                bestNewCen = centerBuff
                bestCluster = clusterBuff.copy()
                bestClassify = sseNotSplit - sseSplit
        bestCluster[np.nonzero(bestCluster[:, 0] == 1)[0], 0] = len(centerList)
        bestCluster[np.nonzero(bestCluster[:, 0] == 0)[0], 0] = bestSplit
        centerList[bestSplit] = bestNewCen[0, :]
        centerList.append(bestNewCen[1, :])
        cluster[np.nonzero(cluster[:, 0] == bestSplit)[0], :] = bestCluster
    return centerList, cluster


filename = r'F:\github\MachineLearning全\input\10.KMeans\testSet2.txt'
dataSet = loadData(filename)
lowestErr = np.inf
bestCenters, cluster = biKmeans(dataSet, 3, distMeans = computeDis)
for i in range(len(bestCenters)):
    bestCenters[i] = bestCenters[i].tolist()[0]
print(bestCenters)
plotDiverse(dataSet, 3, bestCenters, cluster)
