# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 22:47:42 2016

@author: Administrator
"""
from numpy import *
#CART算法的实现代码

def loadDataSet(fileName):
    dataMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        fltLine=map(float,curLine)   #把列表中的每个元素变成浮点数
        dataMat.append(fltLine)
    return dataMat
    
#给定特征切分值将数据分成两部分 
    
def binSplitDataSet(dataSet,feature,value):   #三个参数：数据集合，待切分的特征，该特征的某个值
    mat0=dataSet[nonzero(dataSet[:,feature]>value)[0],:][0] #nonzero(dataSet[:,feature]>value)[0]把大于value的值对应的指针取出
    mat1=dataSet[nonzero(dataSet[:,feature]<=value)[0],:][0]
    return mat0,mat1
    
def regLeaf(dataSet):
    return mean(dataSet[:,-1]) #所有特征的平均值
    
def regErr(dataSet):
    return var(dataSet[:,-1])*shape(dataSet)[0]
#树构建函数   
def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):      
    feat,val=chooseBestSplit(dataSet,leafType,errType,ops)
    if feat==None:
        return val
    retTree={}
    retTree['spInd']=feat
    retTree['spVal']=val
    lSet,rSet=binSplitDataSet(dataSet,feat,val)
    retTree['left']=createTree(lSet,leafType,errType,ops)
    retTree['right']=createTree(rSet,leafType,errType,ops)
    return retTree
    
#回归树的切分函数
def chooseBestSplit(dataSet,leafType,errType,ops=(1,4)):
    tolS=ops[0]
    tolN=ops[1]
    if len(set(dataSet[:,-1].T.tolist()[0]))==1:
        return None,leafType(dataSet)
    m,n=shape(dataSet)
    S=errType(dataSet)
    bestS=inf
    bestIndex=0
    bestValue=0
    for featIndex in range(n-1): #对feature循环
        for splitVal in set(dataSet[:,featIndex]): #对featIndex的所有不同取值循环
            mat0,mat1=binSplitDataSet(dataSet,featIndex,splitVal)
            if(shape(mat0)[0]<tolN)or(shape(mat1)[0]<tolN): #切分的最小样本数tolN
                continue
            newS=errType(mat0)+errType(mat1)
            if newS<bestS:
                bestIndex=featIndex
                bestValue=splitVal
                bestS=newS
    if (S-bestS)<tolS:   #切分减小的最小误差tolS
        return None,leafType(dataSet)
    mat0,mat1=binSplitDataSet(dataSet,bestIndex,bestValue)
    if(shape(mat0)[0]<tolN)or(shape(mat1)[0]<tolN):
         return None,leafType(dataSet)
    return bestIndex,bestValue

#测试回归树函数
def test1():
    myDat=loadDataSet('ex00.txt')
    myMat=mat(myDat)
    RegreTree=createTree(myMat)
    return RegreTree
    
"""chooseBestSplit函数已经做了一定程度的预剪枝，但是条件tolS的不灵活性，有必要继续做后剪枝"""
#回归树剪枝函数
def isTree(obj):
    return (type(obj).__name__=='dict') #注意这个用法来判断obj是否是dict类型
#递归函数，对树进行塌陷处理
def getMean(tree):
    if isTree(tree['right']):
        tree['right']=getMean(tree['right'])
    if isTree(tree['left']):
        tree['left']=getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0 #对叶子节点进行平均，依次返回上层平均，假设对于一个树桩，则不变

def prune(tree,testData): #待剪枝的树tree,剪枝所需要的测试数据
    if shape(testData)[0]==0:
        return getMean(tree)  #没有测试数据，直接对树进行塌陷处理    
    if(isTree(tree['right']) or isTree(tree['left'])): #左右分支有一个任然是子树
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])  
    if isTree(tree['left']):
        tree['left']=prune(tree['left'],lSet)
    if isTree(tree['right']):
        tree['right']=prune(tree['right'],rSet)
    if not isTree(tree['left']) and not isTree(tree['right']): #左右都是叶子结点或者没有结点
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])           #power是幂函数，前面是底数，后面是指数   
        errorNoMerge=sum(power(lSet[:,-1]-tree['left'],2))+sum(power(rSet[:,-1]-tree['right'],2))    #不合并的误差
        treeMean=(tree['left']+tree['right'])/2.0
        errorMerge=sum(power(testData[:,-1]-treeMean,2))     #合并的误差      
        if errorMerge<errorNoMerge:
            print 'merging'
            return treeMean  #合并为一个值，也没有切分点，切分值
        else:
            return tree
    else:
        return tree
#测试后剪枝函数
def test2():
    myDat2=loadDataSet('ex2.txt')
    myMat2=mat(myDat2)
    myTree=createTree(myMat2,ops=(0,1))
    myDatTest=loadDataSet('ex2test.txt')
    myMat2Test=mat(myDatTest)
    tree=prune(myTree,myMat2Test)
    return tree
#模型树的叶节点生成函数
def linearSolve(dataSet):
    m,n=shape(dataSet)
    X=mat(ones((m,n)))
    Y=mat(ones((m,1)))
    X[:,1:n]=dataSet[:,0:n-1]  #注意mat与array的一些区别
    Y=dataSet[:,-1]
    xTx=X.T*X
    if linalg.det(xTx)==0.0:
        raise NameError ('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws=xTx.I*(X.T*Y)
    return ws,X,Y

def modelLeaf(dataSet):
    ws,X,Y=linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y=linearSolve(dataSet)
    yHat=X*ws
    return sum(power(Y-yHat,2))
    
#模型树测试
def test3():
    myMat2=mat(loadDataSet('exp2.txt'))
    tree=createTree(myMat2,modelLeaf,modelErr,(1,10))
    return tree

#树回归进行的代码测试
def regTreeEval(model,inDat):
    return float(model)

def modelTreeEval(model,inDat):
    n=shape(inDat)[1]
    X=mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model) 
#对测试集的每个x计算对应的y
def treeForeCast(tree,inData,modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree,inData)
    if inData[tree['spInd']]>tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)
def createForeCast(tree,testData,modelEval=regTreeEval):
    m=len(testData)
    yHat=mat(zeros((m,1)))     
    for i in range(m):
        yHat[i,0]=treeForeCast(tree,mat(testData[i]),modelEval)
    return yHat        
        
#创建一棵回归树
def test4():
    trainMat=mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat=mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    myTree=createTree(trainMat,ops=(1,20))
    yHat=createForeCast(myTree,testMat[:,0])
    corrcoef1=corrcoef(yHat,testMat[:,1],rowvar=0)[0,1] #corrcoef求出相关系数矩阵，取非对角元素
    return corrcoef1  
#创建一棵模型树
def test5():
    trainMat=mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat=mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    myTree=createTree(trainMat,modelLeaf,modelErr,(1,20))
    yHat=createForeCast(myTree,testMat[:,0],modelTreeEval)
    corrcoef2=corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]
    return corrcoef2
 #标准的线性回归
def test6():
    trainMat=mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat=mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    ws,X,Y=linearSolve(trainMat)
    n=testMat.shape[0]
    yHat=mat(ones((n,1)))
    for i in range(shape(testMat)[0]):
        yHat[i]=testMat[i,0]*ws[1,0]+ws[0,0]
    corrcoef3=corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]
    return corrcoef3
        
