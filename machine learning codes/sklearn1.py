# -*- coding: utf-8 -*-
"""
Created on Tue Dec 06 15:04:49 2016

@author:sklearner
"""
import sklearn
import pylab as pl
import pickle
from sklearn import datasets
from sklearn.externals import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn import random_projection
#SVM模型训练：
clf=svm.SVC()#clf is a estimator
iris=datasets.load_iris()
X,y=iris.data,iris.target
clf.fit(X,y)

#SVM模型存储和测试：
s=pickle.dumps(clf)
clf2=pickle.loads(s)
pre_y=clf2.predict(X[0:1])
if (y[0]==pre_y):
    print 'the prediction is correct'

#joblib处理大数据，可以把clf存储至文件
joblib.dump(clf,'filename.pkl')
clf=joblib.load('filename.pkl')

#Type casting
rng=np.random.RandomState(0)
X=rng.rand(10,2000)
X=np.array(X,dtype='float32')
X.dtype

transformer=random_projection.GaussianRandomProjection()
X_new=transformer.fit_transform(X)
X_new.dtype

#Refitting and updating parameters
rng=np.random.RandomState(0)
X=rng.rand(100,10)
y=rng.binomial(1,0.5,100)
X_test=rng.rand(5,10)
clf=SVC()
clf.set_params(kernel='linear').fit(X,y)#设置核函数为linear
clf.predict(X_test)
clf.set_params(kernel='rbf').fit(X,y)#重新设置核函数为‘rbf’,Guass核函数
clf.predict(X_test)

#绘制一个digits.images的一个图像

digits=datasets.load_digits()
pl.imshow(digits.images[-1],cmap=pl.cm.gray_r)


#kNN
import numpy as np
from sklearn import datasets
iris=datasets.load_iris()
iris_X=iris.data
iris_y=iris.target
np.unique(iris_y)

np.random.seed(0)
indices=np.random.permutation(len(iris_X))#设置0-149的任意排列，随机取样
iris_X_train=iris_X[indices[:-10]]
iris_y_train=iris_y[indices[:-10]]
iris_X_test=iris_X[indices[-10:]]
iris_y_test=iris_y[indices[-10:]]

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(iris_X_train,iris_y_train)
preX=knn.predict(iris_X_test)
preX==iris_y_test
  
#Linear model : from regression to sparsity
diabetes=datasets.load_diabetes()
diabetes_X_train=diabetes.data[:-20]
diabetes_X_test=diabetes.data[-20:]
diabetes_y_train=diabetes.target[:-20]
diabetes_y_test=diabetes.target[-20:]

from sklearn import linear_model
regr=linear_model.LinearRegression()
regr.fit(diabetes_X_train,diabetes_y_train)
print(regr.coef_)
#The mean square error
np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2)#**2向量点乘
#Explain variance score
regr.score(diabetes_X_test,diabetes_y_test)#x与y线性强弱的预测。


#Shrinkage
X=np.c_[.5,1].T
y=[.5,1]
test=np.c_[0,2].T
regr=linear_model.LinearRegression()
import pylab as pl
pl.figure()
np.random.seed(0)#固定随机生成的随机数
for _ in range(6):
    this_X=.1*np.random.normal(size=(2,1))
    regr.fit(this_X,y)
    pl.plot(test,regr.predict(test))
    pl.scatter(this_X,y,s=3)

#sparse method
alphas=np.logspace(-4,-1,6)
regr=linear_model.Lasso()
scores=[regr.set_params(alpha=alpha).fit(diabetes_X_train,diabetes_y_train).score(diabetes_X_test,diabetes_y_test) for alpha in alphas]
best_alpha=alphas[scores.index(max(scores))]
regr.alpha=best_alpha
regr.fit(diabetes_X_train,diabetes_y_train)
print regr.coef_#某些回归系数变为0，达到降维的目的，也可认为是变量选择


#Classification
logistic=linear_model.LogisticRegression(C=1e5)
logistic.fit(iris_X_train,iris_y_train)
logistic.predict(iris_X_test)































