from __future__ import print_function  #为了查看图片大小
from pylab import *
import numpy as np
from PIL import Image
import operator
import matplotlib.pyplot as plt

def unpickle(file):  #加载数据集
    import pickle
    with open(file, 'rb') as fo:
        dictList = pickle.load(fo, encoding='bytes')
    return dictList


def imageToMat(filename):  #读入图片转换为矩阵
    im = Image.open(filename)
    im = im.resize((64, 64))
    im = np.array(im).reshape(1, -1)
    '''
    box = (175, 25, 200, 75) #裁剪指定区域
    region = im.crop(box)
    region = region.transpose(Image.ROTATE_180)
    im.paste(region, box)
    im = im.resize((64, 64))
    '''
    #im.thumbnail((64, 64))  # 创建缩略图
    im = im.reshape(64, 64, 3)
    plt.imshow(im)
    plt.show()
    #print(im.size)  #查看大小


'''
图像存储方式为先存red,然后green, blue，所以要得到原来的图像转置的时候要像下面这样子做。
'''
def plotOnePicture(dataSet):  #将矩阵还原为图片并且显示出来
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    image = dataSet[b'data'][2000]  #这里的b'data'是因为用二进制打开文件的原因，我也不知道怎么去掉！
    image = image.reshape(3, 32, 32).transpose(1,2,0).astype("float") #这里图像的存储方式为每一个像素点的存储，转置以后，第一个像素点就是1 X 3，然后共有32 X 32 X 3可以得到最终的图像。
    plt.imshow(image.astype('uint8'))  #astype做类型转换                           #相当于每一个像素点存储了rgb三个值
    plt.title(dataSet[b'labels'][2000])
    plt.show()


def splitData(dataSet):  #分开数据为数据和标签
    trainMat = dataSet[b'data']
    trainLabel = dataSet[b'labels']
    return trainMat, trainLabel
    

def knnClassify(dataSet, dataLabel, testInv, k):  #数据集，以及待测的数据，以及用k个点进行估计
    diffMat = dataSet - testInv    #这种方式要使用one loop（一个循环）
    diffMatSquare = np.square(diffMat)
    diffSum = np.sum(diffMatSquare, axis = 1)
    distances = np.sqrt(diffSum)
    sortedLabel = distances.argsort()
    classLabel = {}
    for i in range(k):
        buffLabel = int(float(dataLabel[sortedLabel[i]]))
        if buffLabel not in classLabel.keys():
            classLabel[buffLabel] = 0
        classLabel[buffLabel] += 1
    sortedClass = sorted(classLabel.items(), key = operator.itemgetter(1), \
                         reverse = True)
    return sortedClass[0][0]
        
def crossValid(trainSet, trainLabel, k):
    trainLabel = np.mat(trainLabel).T
    numTrain = trainSet.shape[0]
    randIndex = np.random.permutation(numTrain)   #permutation方法，随机排列组合，结果返回array数组，然后下面是数组索引数组
    shuffled_X = trainSet[randIndex, :]
    shuffled_Y = trainLabel[randIndex, :]
    newTrain = shuffled_X[:1600, :]
    newLabel = shuffled_Y[:1600, :]
    newTest = shuffled_X[1600:, :]
    newTestLabel = shuffled_Y[1600:, :]
    errRate = np.zeros(len(k))
    for i in range(len(k)):
        yhat = np.zeros(len(newTest))
        errCount = 0
        for j in range(len(newTest)):
            yhat[j] = knnClassify(newTrain, newLabel, newTest[j], k[i])
            if yhat[j] != newTestLabel[j]:
                errCount += 1
        print('k is %d, and error rate is %.3f' %(k[i], errCount/len(newTest)))
        errRate[i] = errCount/len(newTest)
    minErr = min(errRate)
    bestK = k[int(np.argwhere(errRate == min(errRate)))]  #利用numpy中的argwhere方法找到最小误差的下标
    print('The minimum total error is %.3f, and k is %d' %(minErr, bestK))
    return bestK
    

  
file = r'F:\数据集\斯坦福大学数据集\cifar-10-python\cifar-10-batches-py\data_batch_1'  #其中之一的训练集
labelFile = r'F:\数据集\斯坦福大学数据集\cifar-10-python\cifar-10-batches-py\batches.meta'  #返回的的是下标的英文名字
testFile = r'F:\数据集\斯坦福大学数据集\cifar-10-python\cifar-10-batches-py\test_batch'  #测试集合
dataSet = unpickle(file)   #pickle打开数据集
testSet = unpickle(testFile)
trainSet, trainLabel = splitData(dataSet)   #把数据集分成标签和数据
testSet, testLabel = splitData(testSet)
trainSet = trainSet / 255
testSet = testSet / 255            #归一化
label = unpickle(labelFile)
k_choices = [1, 10, 15, 20, 25]    #利用交叉验证选择合适的k值
bestK = crossValid(trainSet[:2000], trainLabel[:2000], k_choices)   #选取2000张图片交叉验证
errCount = 0
for i in range(len(testLabel[:200])):    #用200张图片做测试
    yhat = knnClassify(trainSet[:2000], trainLabel[:2000], testSet[i], bestK)
    if yhat != testLabel[i]:
        errCount += 1
        print((yhat, testLabel[i]))
print('total error rate is', errCount/len(testLabel[:200]))  #用来其中之一的10000个数据，然后验证总的test，得到最终结果太差劲了,28%的正确率。。。
#print(label[b'label_names'][0] == b'airplane') #举例其中之一的batches.meta文件的label
#plotOnePicture(dataSet)






