
# coding: utf-8

# In[18]:


#导入必要的包
import numpy as np
import matplotlib.pyplot as plt


# In[75]:


def loadDataSet(file):  #加载数据集
    f = open(file)
    retMat = []
    for line in f.readlines():
        lineArr = line.strip().split()
        lineArr = list(map(float, lineArr))
        retMat.append(lineArr)
    return np.mat(retMat)

def pca(dataSet, topNfeat = 99999):  #输入数据集和降维成多少维
    meanVals = np.mean(dataSet, axis = 0)  #计算特征值
    meanRemoved = dataSet - meanVals      #去除均值
    covMat = np.cov(meanRemoved, rowvar = 0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  #将协方差矩阵conMat换成矩阵形式

    #画出特征值变化的图，可以定性的观察出要选择多少特征值来降维
    valChange = eigVals / np.sum(eigVals)
    Ind = valChange.argsort()
    Ind = Ind[::-1]   #转换为逆序数组，因为上面的是从小到大排的
    valChange = valChange[Ind]
    plt.plot(np.arange(20), valChange[:20])  #画出前20个特征向量对应的图
    plt.show()

    eigValSort = eigVals.argsort()
    eigValSort = eigValSort[:-(topNfeat + 1):-1]
    newVects = eigVects[:, eigValSort]   #最大的特征值对应的特征向量
    lowDimVects = meanRemoved * newVects   #计算转换后的特征矩阵
    reconMat = lowDimVects * newVects.T  + meanVals  #降维之后的数据集
    return lowDimVects, reconMat
    


# In[65]:


def plotData(dataSet, pcaData):  #绘制出结果
    plt.subplot(1, 1, 1)
    plt.scatter(dataSet[:, 0].T.tolist()[0], dataSet[:, 1].T.tolist()[0], c = 'green')
    plt.scatter(pcaData[:, 0].T.tolist()[0], pcaData[:, 1].T.tolist()[0], c = 'green')
    plt.show()
    
def nanToMean(dataSet):  #将数据中的nan用平均值代替
    numFeat = dataSet.shape[1]
    for i in range(numFeat):
        meanVal = np.mean(dataSet[np.nonzero(~np.isnan(dataSet[:, i]))[0], i])
        dataSet[np.nonzero(np.isnan(dataSet[:, i]))[0], i] = meanVal   #用平均值代替nan
    return dataSet


# In[76]:


'''  # 例子：半导体制造降维
file = r'E:\jupyter notebook\机器学习实战\pca\testSet.txt'
dataSet = loadDataSet(file)
'''

file = r'E:\jupyter notebook\机器学习实战\pca\secom.data'
dataSet = loadDataSet(file)
print(dataSet.shape)
replaceTOMeanSet = nanToMean(dataSet)
lowDimVects, reconMat = pca(replaceTOMeanSet)

