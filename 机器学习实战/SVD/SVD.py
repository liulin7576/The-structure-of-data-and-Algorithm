
# coding: utf-8

# In[2]:


#导入必要的包
import numpy as np


# In[3]:


a = [[1, 1], [7, 7]]
data = [[1, 1, 1, 0, 0],
       [2, 2, 2, 0 ,0],
       [5, 5, 5, 0, 0],
       [1, 1, 0, 2, 2],
       [0, 0, 0, 3, 3],
       [0, 0, 0, 1, 1]]
U, Sigma, VT = np.linalg.svd(data)
Sig = np.tile(Sigma[:3], (len(Sigma[:3]), 1)) * np.eye(len(Sigma[:3]))  #取最大的3个奇异值
print(np.mat(U[:, :3]) * Sig * VT[:3, :])


# In[4]:


#3种相似度的衡量方式
def euclid(inA, inB):#欧氏距离
    return 1 / (1 + np.linalg.norm(inA - inB))

def pearson(inA, inB):  #pearson相关系数
    if len(inA) < 3:
        return 1
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar = 0)[0][1]

def cosCorr(inA, inB): #计算余弦相似度
    num = float(inA.T * inB)
    denum = np.linalg.norm(inA) * np.linalg.norm(inB)
    return 0.5 + 0.5 * num/denum


# In[5]:


data = np.mat(data)
print('欧氏距离相似度：',euclid(data[:, 0], data[:, 4]))
print('pearson相似度：',pearson(data[:, 0], data[:, 4]))
print('余弦相似度：',cosCorr(data[:, 0], data[:, 4]))


# In[84]:


#基于物品相似度的推荐
'''
思想就是，比如商品1和2，用户没有评价过2，用户评价过1，利用该用户对1的评价分数乘上其他用户在1和2的评价的相似性，来估算用户对于2的喜爱程度。
'''
def standEst(dataMat, user, item, simMeans):   #计算物品item和其他物品的相似度
    n = dataMat.shape[1]
    simTotal, rateSimTotal = 0.0, 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:   #找出用户评过分的商品，没品过分就是要推荐给用户的
            continue
        overLap = np.nonzero(np.logical_and(dataMat[:, item], dataMat[:, j]))[0]  #逻辑与返回两者都大于0的行，即所有用户
        if len(overLap) == 0:
            continue
        similarity = simMeans(dataMat[overLap, item], dataMat[overLap, j])  #计算所有对上述两个物品都评过分的用户计算j 和 item的相似度
        print('the %d and %d similarity is %.3f' %(item, j, similarity))
        simTotal += similarity  #对所有相似度加起来，意思就是考虑未评过分的商品和以评过分的所有商品的相似度
        rateSimTotal += similarity * userRating  #   把相得分转换到0-5之间
    if simTotal == 0:
        return 0
    return rateSimTotal/simTotal

#基于SVD的评分估计
def svdEst(dataMat, user, item, simMeans):  
    U, Sigma, VT = np.linalg.svd(dataMat)
    Sig5 = np.mat(np.tile(Sigma[:5], (5, 1)) * np.eye(5))  #只需要前5个特征向量。然后重构
    newFormedItems = dataMat.T * U[:, :5] * Sig5.I  #不懂
    n = dataMat.shape[1]
    simTotal, rateSimTotal = 0.0, 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:   #找出用户评过分的商品，没品过分就是要推荐给用户的
            continue
        similarity = simMeans(newFormedItems[item, :].T, newFormedItems[j, :].T)
        print('the %d and %d similarity is %.3f' %(item, j, similarity))
        simTotal += similarity  #对所有相似度加起来，意思就是考虑未评过分的商品和以评过分的所有商品的相似度
        rateSimTotal += similarity * userRating  #   把得分转换到0-5之间
    if simTotal == 0:
        return 0
    return rateSimTotal/simTotal


def reommend(dataMat, user, N = 3, simMeans = cosCorr, estMethod = standEst):  #推荐函数
    unRatedItems = np.nonzero(dataMat[user, :] == 0)[1]  #找出用户没有评级的商品
    if len(unRatedItems) == 0:
        return 'All goods are rated!'
    itemScore = []
    for item in unRatedItems:
        estimatedScore = estMethod(dataMat, user, item, simMeans)
        itemScore.append((item, estimatedScore))
    return sorted(itemScore, key = lambda p: p[1], reverse = True)[:N]





# In[102]:


#简单的图像压缩
def printMat(inMat, thresh = 0.8):  #因为用奇异值分解以后的矩阵有小数，需要转换为0,1
    n = inMat.shape[1]
    for i in range(n):
        buff = []
        for j in range(n):
            if inMat[i, j] > thresh:
                buff.append(1)
            else:
                buff.append(0)
        print(buff)
    return inMat

def imageCompress(numSV = 3, thresh = 0.8): #图像压缩
    f = open(r'F:\github\MachineLearning全\input\14.SVD\0_5.txt')
    imageMat = []
    for line in f.readlines():
        lineArr = []
        for j in range(32):
            lineArr.append(int(line[j]))
        imageMat.append(lineArr)
    imageMat = np.mat(imageMat)
    print('*********** original image *************')
    printMat(imageMat)
    U, Sigma, VT = np.linalg.svd(imageMat)
    sigSV = np.tile(Sigma[:numSV], (numSV, 1)) * np.eye(numSV)
    formedMat = U[:, :numSV] * sigSV * VT[:numSV, :]
    print('*************** After compress ************')
    printMat(formedMat, thresh)

    f.close()

    
#调用函数尝试一下
imageCompress(numSV = 3, thresh = 0.8)


# In[100]:


dataSet = [[2, 0, 0, 4, 4, 0, 0 ,0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
          [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
          [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
          [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
          [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
          [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
          [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
          [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
          [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]
dataSet = np.mat(dataSet)
'''  #计算多少维特征的和相加满足大于0.9
U, Sigma, VT = np.linalg.svd(dataSet)
Sigma = Sigma ** 2
totalVal = np.sum(Sigma)
for item in range(len(Sigma)):
    if np.sum(Sigma[:(item + 1)])/totalVal >= 0.9:
        print('the item is', item + 1)
        break
'''
reommend(dataSet, 10, N = 3,simMeans = cosCorr, estMethod = standEst)


# In[87]:


#验证上述的函数
dataMat = np.mat(data)
print(dataMat)
reommend(dataMat, 2, N = 1, simMeans = pearson, estMethod = standEst)

