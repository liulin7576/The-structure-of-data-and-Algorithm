import matplotlib
import matplotlib.pyplot as  plt

def grabTree(filename): #将树取出来
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)
decisionNode = dict(boxstyle = 'sawtooth', fc = '0.8')
leafNode = dict(boxstyle = 'round4', fc = '0.8')
arrow_args = dict(arrowstyle = '<-')


def plotNode(nodeTxt, centerPt, parentPt, nodeType):#一堆画图参数，到时再查把
    createPlot.ax1.annotate(nodeTxt, xy = parentPt,\
    xycoords = 'axes fraction', xytext = centerPt, textcoords = 'axes fraction',\
    va = 'center', ha = 'center', bbox = nodeType, arrowprops = arrow_args)


def getNumLeafs(myTree): #得到叶子节点数目
    numLeafs = 0
    firstStr = list(myTree.keys())[0] #得到特征对应的key值
    secondDict = myTree[firstStr]
    for k in secondDict:
        if type(secondDict[k]).__name__ == 'dict':##这里的if语句一定要使用.__name__
            numLeafs += getNumLeafs(secondDict[k]) #递归遍历
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree): # 得到数的深度
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for k in secondDict:
        if type(secondDict[k]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[k])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def plotMidText(cntrPt, parentPt, txtString): #在中心点上添加标注
    xMid = (parentPt[0] - cntrPt[0]) / 2 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt): #画树
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xoff + (1.0 + float(numLeafs))/2/plotTree.totalW,\
              plotTree.yoff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yoff -= 1.0 / plotTree.totalD
    for key in secondDict:
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xoff += 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xoff, plotTree.yoff),\
                     cntrPt, leafNode)
            plotMidText((plotTree.xoff, plotTree.yoff), cntrPt, str(key))
    plotTree.yoff += 1.0 / plotTree.totalD #加的原因是遍历一颗子树结束以后，要返回上一层，那么需要加1

def createPlot(inTree): #画图实现
    fig = plt.figure(1, facecolor = 'white')
    fig.clf()
    axprops = dict(xticks = [], yticks = []) # 去除坐标值的显示
    createPlot.ax1 = plt.subplot(111, frameon = False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xoff = -0.5 / plotTree.totalW;plotTree.yoff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

filename = r'E:\python编程处\机器学习实战\决策树\storeTree.txt'
myTree = grabTree(filename)
createPlot(myTree)

