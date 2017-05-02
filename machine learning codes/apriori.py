# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 18:30:18 2016

@author: Administrator
"""
#Apriori算法中的辅助函数
def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet):
    C1=[]
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return map(frozenset,C1)  #把C1中的每个元素都变成frozenset的形式

def scanD(D,Ck,minSupport):
    ssCnt={}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can):  #判断can是否是ssCnt的键
                    ssCnt[can]=1
                else:
                    ssCnt[can]+=1
    numItems=float(len(D))
    retList=[]     #取出支持度大于minsupport的项集
    supportData={}  #用来存储retLiset中各项的支持度
    for key in ssCnt:
        support=ssCnt[key]/numItems
        if support >=minSupport:
            retList.insert(0,key)   #在列表0处插入key
        supportData[key]=support
    return retList,supportData
    
#Apriori算法
def aprioriGen(Lk,k):
    retList=[]
    lenLk=len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1=list(Lk[i])[:k-2]
            L2=list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1==L2:       #两个项集前k-2个项相同时，两个集合并
                retList.append(Lk[i]|Lk[j])
    return retList
def apriori(dataSet,minSupport=0.5):
    C1=createC1(dataSet) #dataSet中的所有单个物品项
    D=map(set,dataSet)  
    L1,supportData=scanD(D,C1,minSupport)#最频繁集L1，和对应项的支持度supportData
    L=[L1]
    k=2
    while (len(L[k-2])>0):
        Ck=aprioriGen(L[k-2],k)
        Lk,supk=scanD(D,Ck,minSupport)
        supportData.update(supk)
        L.append(Lk)
        k+=1
    return L,supportData
    
#测试函数
def test():
    dataSet=loadDataSet()
    L,suppData=apriori(dataSet,minSupport=0.7)
    print L[0],L[1],L[2]   
    print suppData
    
#关联规则生成函数
def generateRules(L,supportData,minConf=0.7):  #频繁项集列表，包含频繁项集列表支持数据的字典，最小可信度阈值
    bigRuleList=[]
    for i in range(1,len(L)):
        for freqSet in L[i]:
            H1=[frozenset([item]) for item in freqSet]
            if (i>1):
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
            else:
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList
 
def calcConf(freqSet,H,supportData,brl,minConf=0.7):
    prunedH=[]
    for conseq in H:
        conf=supportData[freqSet]/supportData[freqSet-conseq]   #计算可信度，可信度为（freqSet-conseq）---->（conseq）
        if conf>=minConf:
            print freqSet-conseq,'-->',conseq,'conf:',conf
            brl.append((freqSet-conseq,conseq,conf))  #存储关联规则的列表
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet,H,supportData,brl,minConf=0.7):
    m=len(H[0])
    if (len(freqSet))>(m+1):
        Hmp1=aprioriGen(H,m+1)
        Hmp1=calcConf(freqSet,Hmp1,supportData,brl,minConf)
        if(len(Hmp1)>1):
            rulesFromConseq(freqSet,Hmp1,supportData,brl,minConf)
        
#测试函数
def test2():
    dataSet=loadDataSet()
    L,suppData=apriori(dataSet,minSupport=0.5)
    rules=generateRules(L,suppData,minConf=0.7)
    return rules
    