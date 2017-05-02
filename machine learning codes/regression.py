# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 15:15:42 2016

@author: Administrator
"""
from numpy import *
#标准回归函数和数据导入函数
def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))-1
    dataMat=[];labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat=mat(xArr);yMat=mat(yArr).T
    xTx=xMat.T*xMat
    if linalg.det(xTx)==0.0:#linalg.det计算矩阵的行列式
        print "This matrix is singular,cannot do inverse"
        return
    ws=xTx.I*(xMat.T*yMat)
    return ws

#绘制图像
def plotIt():
    xArr,yArr=loadDataSet('ex0.txt')
    ws=standRegres(xArr,yArr)
    xMat=mat(xArr)
    yMat=mat(yArr)
    yHat=xMat*ws
    import matplotlib.pyplot as plt
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
    
    xCopy=xMat.copy()
    xCopy.sort(0)
    yHat=xCopy*ws
    ax.plot(xCopy[:,1],yHat)
    
    corr=corrcoef(yHat.T,yMat)
    print 'the correlation of y and yhat is ',corr

#局部加权线性回归函数
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat=mat(xArr);yMat=mat(yArr).T
    m=shape(xMat)[0]
    weights=mat(eye((m)))
    for j in range(m):
        diffMat=testPoint-xMat[j,:]
        weights[j,j]=exp(diffMat*diffMat.T/(-2*k**2))
    xTx=xMat.T*(weights*xMat)
    if linalg.det(xTx)==0:
        print 'This matrix is singula,cannot do inverse'
        return
    ws=xTx.I*(xMat.T*(weights*yMat))
    return testPoint*ws
    
def lwlrTest(testArr,xArr,yArr,k=1.0):
    m=shape(testArr)[0]
    yHat=zeros(m)
    for i in range(m):
        yHat[i]=lwlr(testArr[i],xArr,yArr,k)
    return yHat
#绘制图像（2）
def plotIts():
    xArr,yArr=loadDataSet('ex0.txt')
    yHat=lwlrTest(xArr,xArr,yArr,0.003)
    xMat=mat(xArr)
    srtInd=xMat[:,1].argsort(0)
    xSort=xMat[srtInd][:]#特别注意得到的特殊的三维矩阵
    import matplotlib.pyplot as plt
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s=2,c='red')
    plt.show()

#示例：预测鲍鱼的年龄
#求残差平方和
def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()    
#观察shrinkage的效果   
def test():
    a,b=loadDataSet('abalone.txt')
    ridgeWeights=ridgeTest(a,b)
    import matplotlib.pyplot as plt
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()
    
#岭回归
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx=xMat.T*xMat
    denom=xTx+eye(shape(xMat)[1])*lam
    if linalg.det(denom)==0.0:
        print 'This matrix is singular,cannot do inverse'
        return
    ws=denom.I*(xMat.T*yMat)
    return ws

def ridgeTest(xArr,yArr):
    xMat=mat(xArr);yMat=mat(yArr).T#转为一行的矩阵
    yMean=mean(yMat,0)
    yMat=yMat-yMean
    xMeans=mean(xMat,0)
    xVar=var(xMat,0)
    xMat=(xMat-xMeans)/xVar
    numTestPts=30
    wMat=zeros((numTestPts,shape(xMat)[1]))  
    for i in range(numTestPts):
        ws=ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i:]=ws.T
    return wMat  #不同的30个lamda取值下的对应的回归系数W的值
    
#前向逐步线性回归
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m,n=shape(xMat)
    returnMat = zeros((numIt,n)) #testing code remove
    ws = zeros((n,1))
    for i in range(numIt):#could change this to while loop
        #print ws.T
        lowestError = inf; 
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

#数据标准化函数
def regularize(xMat):#regularize by columns  
    inMat = xMat.copy()  
    inMeans = mean(inMat,0)   #calc mean then subtract it off  
    inVar = var(inMat,0)      #calc variance of Xi then divide by it  
    inMat = (inMat - inMeans)/inVar    
    return inMat  
    
    
    