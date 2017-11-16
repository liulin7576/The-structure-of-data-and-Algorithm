def simpleData():  #示例程序，简单的数据集
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):  #构建所有项的集合,frozenset还是set都可以，在后面用字典的时候转换为frozenset就可以了
    C1 = []
    for i in dataSet:
        for j in i:
            if [j] not in C1:
                C1.append([j])
    C1.sort()
    return map(frozenset, C1)  #返回的是一个迭代器对象，所以遍历一次就结束了，需要用list()转换为列表就好了，切记切记

def loadDataSet(filename):  #加载毒蘑菇数据
    f = open(filename)
    dataSet = []
    for line in f.readlines():
        lineArr = line.strip().split()
        dataSet.append(lineArr)
    return dataSet

    

def scanD(dataSet, Ck, minSupport):   #计算并返回大于最小支持度的项集
    ssCnt = {}
    numItems = len(dataSet)
    for tran in dataSet:
        for can in Ck:   #前面把frozenset变成了列表，不然这里没法多次遍历，因为是map对象
            if can.issubset(tran):
                can = frozenset(can)  #这里一定要把set转换为frozenset，不然字典没法处理set，但是字典可以处理frozenset
                if can not in ssCnt.keys():
                    ssCnt[can] = 0
                ssCnt[can] += 1
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key) #z在首端插入key
        supportData[key] = support
    return supportData, retList

def aprioriGen(Lk, k):  #将小的项集合并为多的项集
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk - 1):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]  #用k-2是为了更快的遍历，得到更高的项集可以这样子做
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

         
def apriori(dataSet, minSupport = 0.5):   # 给定dataSet和支持率，然后得到所有满足支持率的项集
    C1 = list(createC1(dataSet))  #what? 加了一个list就可以了？之前frozenset创建的集合永远只能遍历一次，好奇怪！现在懂了！
    supportData, L1 = scanD(dataSet, C1, minSupport)
    L = [L1]
    k = 2
    while len(L[k - 2]) > 0:
        Ck = aprioriGen(L[k - 2], k) #合并项集
        supK, Lk = scanD(dataSet, Ck, minSupport)  #踢出不满足支持度的项集
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

def generateRules(L, supportData, minConf = 0.7): #计算关联规则
    bigRuleList = []   #用于构造最后的关联列表
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]  #这里的[item]一定要加上括号，因为是集合
            if i > 1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def calcConf(freqSet, H, supportData, brl, minConf = 0.7): #计算可信度
    confH = []
    for conSeq in H:
        conf = supportData[freqSet] / supportData[freqSet - conSeq]
        if conf >= minConf: #剔除不满足最小可信度的项，即假如1->023不成立，那么10->23更不成立了！所以不需要保留1
            brl.append((freqSet - conSeq, conSeq, conf))
            #print(freqSet - conSeq, '-->', conSeq, 'conf:',conf)
            confH.append(conSeq)
    return confH


def rulesFromConseq(freqSet, H, supportData, brl, minConf = 0.7):  #尝试进一步合并
    m = len(H[0]) #开始时候的H[0]=1，然后第二次循环为2，依次...
    if len(freqSet) > (m + 1): #这里大于m+1的原因是如果等于m+1，那么下面的aprioriGen就会产生和freqSet一样长的set,然后计算calcConf的时候，分母就会为空集
        Hmp1 = aprioriGen(H, m + 1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)   #把满足条件的项抽取出来,输入Hmp1,输出满足最小可信度的Hmp1
        if len(Hmp1) > 1: 
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)





filename = r'F:\github\MachineLearning全\input\11.Apriori\mushroom.dat'
dataSet = loadDataSet(filename)
L, supportData = apriori(dataSet, minSupport = 0.5)
for i in L[1]:
    if i.intersection('2'):  #intersection是set集合有的方法，且适用于元素为字符串。有该字符串，直接返回该字符串。
        print(i)

















    
