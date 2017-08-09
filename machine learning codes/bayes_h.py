# -*- coding: utf-8 -*-
"""贝叶斯分类器"""
from sklearn import datasets, cross_validation, naive_bayes
import numpy as np
import matplotlib.pyplot as plt


# 载入digit数据集
def show_digits():
    digits = datasets.load_digits()
    fig = plt.figure()
    print "vector from images 0", digits.data[0]
    for i in range(25):
        ax = fig.add_subplot(5, 5, i + 1)  # 把fig分成5行5列，在从左到右从上到下数位于第i位置上增加子图
        ax.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()


# 加载数据集的函数
def load_data():
    digits = datasets.load_digits()
    return cross_validation.train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)


# 测试高斯贝叶斯分类器的函数
def test_GaussianNB(*data):
    X_train, X_test, y_train, y_test = data
    cls = naive_bayes.GaussianNB()
    cls.fit(X_train, y_train)
    print "Training Score:%.2f" % cls.score(X_train, y_train)
    print "Testing Score:%.2f" % cls.score(X_test, y_test)


data = load_data()
test_GaussianNB(*data)


# 测试多项式贝叶斯分类器的函数
def test_MultinomialNB(*data):
    X_train, X_test, y_train, y_test = data
    cls = naive_bayes.MultinomialNB()
    cls.fit(X_train, y_train)
    print "Training Score:%.2f" % cls.score(X_train, y_train)
    print "Testing Score:%.2f" % cls.score(X_test, y_test)


data = load_data()
test_MultinomialNB(*data)

# 不同的alpha对多项式贝叶斯分类器的影响
def test_MultinomialNB_alpha(*data):
    X_train, X_test, y_train, y_test = data
    alphas = np.logspace(-2,5,num=200)
    train_scores=[]
    test_scores=[]
    for alpha in alphas:
        cls=naive_bayes.MultinomialNB(alpha=alpha)
        cls.fit(X_train,y_train)
        train_scores.append(cls.score(X_train,y_train))
        test_scores.append(cls.score(X_test,y_test))
    ## plot
    fig =plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(alphas,train_scores,label="Training Score")
    ax.plot(alphas,test_scores,label="Testing Score")
    ax.set_xlabel("alpha")
    ax.set_ylabel("score")
    ax.set_ylim(0,1.0)
    ax.set_title("MultinomialNB")
    ax.set_xscale("log")
    plt.show()
data=load_data()
test_MultinomialNB_alpha(*data)

# 测试伯努利分类器的函数
def test_BernoulliNB(*data):
    X_train, X_test, y_train, y_test = data
    cls=naive_bayes.BernoulliNB()
    cls.fit(X_train,y_train)
    print "Training Score:%.2f" % cls.score(X_train, y_train)
    print "Testing Score:%.2f" % cls.score(X_test, y_test)
data=load_data()
test_BernoulliNB(*data)

# 递增式学习 partial_fit方法