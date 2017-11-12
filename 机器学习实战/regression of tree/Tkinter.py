import regTree  #导入之前的regTree包
from tkinter import *
import numpy as np
import matplotlib
from matplotlib.figure import Figure
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def reDraw(tolS, tolN):  #画出训练模型的图形
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(1, 1, 1)
    if chkBtnVar.get():
        if tolN < 2:
            tolN =2
        myTree = regTree.createTree(reDraw.rawDat, regTree.modelLeaf, regTree.modelErr, (tolS, tolN))
        yhat = regTree.apiModelTest(myTree, np.mat(reDraw.testDat).T)
    else:
        myTree = regTree.createTree(reDraw.rawDat, regTree.regLeaf, regTree.regErr, (tolS, tolN))
        yhat = regTree.apiRegTest(myTree, np.mat(reDraw.testDat).T)
    reDraw.a.scatter(list(map(float, reDraw.rawDat[:, 0])), reDraw.rawDat[:, 1].T.tolist()[0], s = 5)  #这里的预测数据原来是mat矩阵，需要转换为列表
    reDraw.a.plot(reDraw.testDat, yhat, linewidth = 2.0)
    reDraw.canvas.show()

        
def drawNewTree():
    tolS, tolN = getInputs()
    reDraw(tolS, tolN)

def getInputs():   #判断用户是否按下重新开始画图的键
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print('enter integer for tolN')
        tolNentry.delete(0, END)
        tolN.insert(0, '10')
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print('enter float for tolS')
        tolSentry.delete(0, END)
        tolS.insert(0, '1.0')
    return tolS, tolN


root = Tk()

Label(root, text = 'plot place holder').grid(row = 0, columnspan = 3)
reDraw.f = Figure(figsize = (5, 4), dpi = 100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master = root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row = 0, columnspan = 3)

Label(root, text = 'tolN').grid(row = 1,  column = 0)
tolNentry = Entry(root)
tolNentry.grid(row = 1, column = 1)
tolNentry.insert(0, '10')
Label(root, text = 'tolS').grid(row = 2, column = 0)
tolSentry = Entry(root)
tolSentry.grid(row = 2, column = 1)
tolSentry.insert(0, '1.0')
Button(root, text = 'ReDraw', command = drawNewTree).grid(row = 1, column = 2,\
                                                          rowspan = 3)

chkBtnVar = IntVar()
chkBtn = Checkbutton(root, text = 'Model Tree', variable = chkBtnVar)
chkBtn.grid(row = 3, column = 0, columnspan = 2)

reDraw.rawDat = regTree.loadData(r'F:\github\MachineLearning全\input\9.RegTrees\sine.txt')
reDraw.testDat = np.arange(min(reDraw.rawDat[:, 0]),\
                           max(reDraw.rawDat[:, 0]), 0.01)

reDraw(1.0, 10)

root.mainloop()

