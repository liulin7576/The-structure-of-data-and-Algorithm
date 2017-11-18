
# coding: utf-8

# In[44]:


#Fp树的定义
class treeNode:  #定义一个FP树类
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.parent = parentNode
        self.children = {}
        self.nodeLink = None
        
    def inc(self, numOccur):  #对count变量增加给定值
        self.count += numOccur
        
    def disp(self, ind = 1):  #将树以文本形式给出,ind的作用是模仿树，有空一格的作用
        print(' '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)


# In[45]:


#一些验证例子
'''
rootNode = treeNode('pyramid', 9, None)
childNode = treeNode('eye', 13, None)
rootNode.children['eye'] = childNode
rootNode.children['phoenix'] = treeNode('phoenix', 9, None)
childNode.children['thirdTree'] = treeNode('thirdTree', 9, None)
rootNode.disp()
'''


# In[46]:


def createTree(dataSet, minSup = 1):  #创造FP树
    headTable = {}
    for trans in dataSet:
        for item in trans:
            headTable[item] = headTable.get(item, 0) + dataSet[trans]
   #奇怪，下面的方法还是报错，无奈，只有换方法了！
   # for k in headTable.keys():
   #     if headTable[k] < minSup:
   #        del(headTable[k])
    
    buff = []
    for k in headTable.keys():
        if headTable[k] < minSup:
            buff.append(k)
    for k in buff:       
        del(headTable[k])
        
    freqItemSet = set(headTable.keys())#频繁项集    
    if len(freqItemSet) == 0:
        return None, None
    for k in headTable:
        headTable[k] = [headTable[k], None]  #保存名字及出现次数，并且扩展到存储指向下一个的nodeLink指针 
    retTree = treeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headTable[item][0]
        if len(localD) > 0:
            orderItem = [v[0] for v in sorted(localD.items(), key = lambda p: p[1], reverse = True)]
            updataTree(orderItem, retTree, headTable, count)
    return retTree, headTable


def updataTree(item, inTree, headTable, count):  #更新Fp树，即增加结点
    if item[0] in inTree.children.keys():
        inTree.children[item[0]].inc(count)
    else:
        inTree.children[item[0]] = treeNode(item[0], count, inTree)
        if headTable[item[0]][1] == None:
            headTable[item[0]][1] = inTree.children[item[0]]
        else:
            updateHead(headTable[item[0]][1], inTree.children[item[0]])

    if len(item) > 1:
        updataTree(item[1:], inTree.children[item[0]], headTable, count)
    
def updateHead(nodeToTest, lastNode):  #连接子结点到最后一个结点
    while nodeToTest.nodeLink != None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = lastNode
        
    


# In[47]:


def loadSimpleData():  #人工创造一个简单的数据
    simpleData = [['r', 'z', 'h', 'j', 'p'],
                 ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                 ['z'],
                 ['r', 'x', 'n', 'o', 's'],
                 ['z', 'y', 'r', 'x',  'q', 't', 'p'],
                 [ 'z', 'y', 'x', 'e', 'q', 's', 't', 'm']]
    return simpleData

def initDataSet(dataSet):  #初始化数据
    retDict = {}
    for item in dataSet:
        retDict[frozenset(item)] = 1
    return retDict


# In[48]:


#测试上面的代码
dataSet = loadSimpleData()
dataSet = initDataSet(dataSet)
retTree, headTable = createTree(dataSet, minSup = 3)
#retTree.disp()


# In[51]:


def ascendTree(leafNode, prePath):  #利用前面的parent指针向上遍历整棵树
    while leafNode.parent != None:
        prePath.append(leafNode.name)
        leafNode = leafNode.parent

def findPrePath(baseSet, headNode):  #利用前面的headTable保留的指针向上遍历树
    retDict = {}
    while headNode != None:
        prePath = []
        ascendTree(headNode, prePath)
        if len(prePath) > 1:
            retDict[frozenset(prePath[1:])] = headNode.count
        headNode = headNode.nodeLink
    return retDict


# In[52]:


#prePathSet = findPrePath('s', headTable['s'][1])
#print(prePathSet)


# In[79]:


def mineTree(inTree, headTable, minSup, preFix, freqItemList):  #找出频繁项集
    bigL = [v[0] for v in sorted(headTable.items(), key = lambda p: p[1][0])]
    
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPrePath = findPrePath(basePat, headTable[basePat][1])
        myCondTree, myHead = createTree(condPrePath, minSup)

        if myHead != None:
            '''
            print('当前的条件树为：',basePat)
            myCondTree.disp()
            '''
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)


def loadDataSet(file):  #打开100万条记录的文本文件
    f = open(file)
    retMat = []
    for line in f.readlines():
        retMat.append(line.strip().split())
    f.close()
    return retMat


file = r'E:\python编程处\机器学习实战\Fp-growth\kosarak.dat'
parseSet = loadDataSet(file)
initSet = initDataSet(parseSet)
retTree, headTable = createTree(initSet, minSup = 100000)
freqItems = []
mineTree(retTree, headTable, 100000, set([]), freqItems)
print(freqItems)





'''  #发现没法运行，真的是
import twitter
from time import sleep
import re
def getLotsOfTweets(searchStr):   #处理认证
    CONSUMER_KEY = 'get when you create an app'
    CONSUMER_SECRET = 'get when you create an app'
    ACCESS_TOKEN_KEY = 'get from Oauth, specific to a user'
    ACCESS_TOKEN_SECRET = 'get from Oauth, specific to a user'
    api = twitter.Api(consumer_key = CONSUMER_KEY,
                      consumer_secret = CONSUMER_SECRET,
                      access_token_key = ACCESS_TOKEN_KEY,
                      access_token_secret = ACCESS_TOKEN_SECRET)
    resultsPages = []
    for i in range(1, 15):
        peinr('fetching page %d' %i)
        searchResults = api.GetSearch(searchStr, per_page = 100, page = i)
        sleep(5)
    return resultsPages


def storeTweet(tweet, file):  #存储为pickle格式
    import pickle
    fw = open(r'file', 'wb')
    pickle.dump(tweet, fw)
    fw.close()

def grabTweet(file):
    import pickle
    fr = open(r'file', 'rb')
    return pickle.load(fr)
    
file = 'E:\python编程处\机器学习实战\Fp-growth\storeTweet.txt'
tweet = getLotsOfTweets('RIMM')
storeTweet(tweet, file)
'''

# In[80]:

'''
freqItems = []
mineTree(retTree, headTable, 3, set([]), freqItems)
print(freqItems)
'''

