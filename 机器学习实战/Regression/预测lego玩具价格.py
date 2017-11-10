import numpy as np
import random
import time
from bs4 import BeautifulSoup   #要安装爬虫bs4文件，官网下载即可
random.seed(time.time())


def scrapePage(retX, retY, inFile, yr, numPce, origPrc):

    # 打开并读取HTML文件
    fr = open(inFile, 'rb')
    soup = BeautifulSoup(fr.read())
    i = 1

    # 根据HTML页面结构进行解析
    currentRow = soup.findAll('table', r="%d" % i)
    while(len(currentRow)!=0):
        currentRow = soup.findAll('table', r="%d" % i)
        title = currentRow[0].findAll('a')[1].text
        lwrTitle = title.lower()

        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0

        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
        if len(soldUnicde)==0:
            print ("item #%d did not sell" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].findAll('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$','') #strips out $
            priceStr = priceStr.replace(',','') #strips out ,
            if len(soldPrice)>1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)

            # 去掉不完整的套装价格
            if  sellingPrice > origPrc * 0.5:
                    #print( "%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        i += 1
        currentRow = soup.findAll('table', r="%d" % i)

# 依次读取六种乐高套装的数据，并生成数据矩阵        
def setDataCollect(retX, retY):
    scrapePage(retX, retY, 'F:/github/MachineLearning全/input/8.Regression/setHtml/lego8288.html', 2006, 800, 49.99)
    scrapePage(retX, retY, 'F:/github/MachineLearning全/input/8.Regression/setHtml/lego10030.html', 2002, 3096, 269.99)
    scrapePage(retX, retY, 'F:/github/MachineLearning全/input/8.Regression/setHtml/lego10179.html', 2007, 5195, 499.99)
    scrapePage(retX, retY, 'F:/github/MachineLearning全/input/8.Regression/setHtml/lego10181.html', 2007, 3428, 199.99)
    scrapePage(retX, retY, 'F:/github/MachineLearning全/input/8.Regression/setHtml/lego10189.html', 2008, 5922, 299.99)
    scrapePage(retX, retY, 'F:/github/MachineLearning全/input/8.Regression/setHtml/lego10196.html', 2009, 3263, 249.99)


def standReg(trainSet, trainLabel): #标准的最小二乘回归求解
    XTX = trainSet.T * trainSet
    if np.linalg.det(XTX) == 0:
        print('It is no inverse!')
        return
    wHat = XTX.I * (trainSet.T * trainLabel)
    return wHat

def ridgeReg(trainSet, trainLabel, lambd = 0.2):  #岭回归
    XTX = (trainSet.T * trainSet + lambd * np.eye(trainSet.shape[1]))
    if np.linalg.det(XTX) == 0:
        print('No inverse')
        return
    wHat = XTX.I * trainSet.T * trainLabel  #这里的求逆矩阵怎么每次都忘了，真的是
    return wHat

def ridgeTest(trainSet, trainLabel):  #测试岭回归
    yMean = np.mean(trainLabel, 0)
    yMat = trainLabel - yMean
    xMean = np.mean(trainSet, 0)
    xVar = np.var(trainSet, 0)
    xMat = (trainSet - xMean)/xVar
    numIter = 30
    wMat = np.zeros((numIter, trainSet.shape[1]))
    for i in range(numIter):
        wHat = ridgeReg(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = wHat.T
    return np.mat(wMat), yMean, xMean, xVar


def dataLoad():  #加载数据
    retX, retY = [], []
    setDataCollect(retX, retY)
    retX1, retY1 = np.mat(retX), np.mat(retY).T  
    m, n = retX1.shape
    finalX = np.mat(np.zeros((m, n+1)))    #finalX是计算新的输入样本的时候，不需要进行转换直接可以求解
    finalX[:, 0] = np.ones((m, 1))
    finalX[:, 1:n+1] = retX1
    return retX, retY, finalX

def crossValid(numVal = 10): #进行交叉验证
    retX, retY, finalX = dataLoad() #其中finalX是指加第一列为1的数据
    m = len(retY)
    indexList = [i for i in range(m)]
    errMat = np.zeros((m, 30))  #用30的原因是后面的ridgeTest中用的是30个模型
    for i in range(numVal):
        trainX, trainY = [], []
        testX, testY = [], []
        random.shuffle(indexList)
        for j in range(m):
            if j < m*0.9:
                trainX.append(retX[indexList[j]])
                trainY.append(retY[indexList[j]])
            else:
                testX.append(retX[indexList[j]])
                testY.append(retY[indexList[j]])
        trainX, trainY = np.mat(trainX), np.mat(trainY).T
        testX, testY = np.mat(testX), np.mat(testY).T
        wMat, yMean, xMean, xVar = ridgeTest(trainX, trainY)
        for k in range(30):
            testX = (testX - xMean)/xVar
            yhat = testX * wMat[k].T + yMean
            errMat[i, k] = np.sum(np.square(yhat, testY))
    meanErr = np.mean(errMat, 0)
    lowestErr = min(meanErr)
    bestWeight = wMat[np.nonzero(meanErr == lowestErr)]
    meanX, meanY = np.mean(retX, 0), np.mean(retY, 0)
    varX = np.var(retX, 0)
    unReg = bestWeight/varX
    constant = -1 * np.sum(np.multiply(unReg, meanX)) + yMean
    print('final parameter is',[constant, unReg])
    print('before transforming, bestParameters is', bestWeight)
        




if __name__ == '__main__':
    crossValid()
























