# -*- coding: utf-8 -*-

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import cross_validation
import matplotlib.pyplot as plt


# 随机产生的数据集
def creat_data(n):
    np.random.seed(0)
    X = 5 * np.random.rand(n, 1)  # 保证每次生成的随机数相同
    y = np.sin(X).ravel()
    noise_num = (int)(n / 5)
    y[::5] += 3 * (0.5 - np.random.rand(noise_num))  # y数据每隔五个点加一个噪声
    return cross_validation.train_test_split(X, y, test_size=0.25, random_state=1)


# 回归决策树测试函数
def test_DescisionTreeRegressor(*data):
    X_train, X_test, y_train, y_test = data
    regr = DecisionTreeRegressor()
    regr.fit(X_train, y_train)
    print "Training score:%f" % (regr.score(X_train, y_train))
    print "Testing score:%f" % (regr.score(X_test, y_test))
    ## plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    X = np.arange(0.0, 5.0, 0.1)[:, np.newaxis]  # 等价于.reshape(1,50)
    Y = regr.predict(X)
    ax.scatter(X_train, y_train, label="train sample", c='g')
    ax.scatter(X_test, y_test, label="test sample", c='r')
    ax.plot(X, Y, label="predict_value", linewidth=2, alpha=0.5)
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    ax.set_title("Decision Tree Regression")
    ax.legend(framealpha=0.5, loc="best")
    plt.show()


data = creat_data(100)
test_DescisionTreeRegressor(*data)


# 比较最优划分和随机划分的区别
def test_DecisionTreeRegressor_splitter(*data):
    X_train, X_test, y_train, y_test = data
    splitters = ['best', 'random']
    for splitter in splitters:
        regr = DecisionTreeRegressor(splitter=splitter)
        regr.fit(X_train, y_train)
        print "Splitter %s" % splitter
        print "Training score:%f" % (regr.score(X_train, y_train))
        print "Testing score:%f" % (regr.score(X_test, y_test))


data = creat_data(100)
test_DecisionTreeRegressor_splitter(*data)


# 比较不同深度树的区别
def test_DecisionTreeRegressor_depth(maxdepth, *data):
    X_train, X_test, y_train, y_test = data
    depths = np.arange(1, maxdepth)
    training_scores = []
    testing_scores = []
    for depth in depths:
        regr = DecisionTreeRegressor(max_depth=depth)
        regr.fit(X_train, y_train)
        training_scores.append(regr.score(X_train, y_train))
        testing_scores.append(regr.score(X_test, y_test))
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(depths, training_scores, label="training score")
    ax.plot(depths, testing_scores, label="testing score")
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Regression")
    ax.legend(framealpha=0.5, loc="best")
    plt.show()


data = creat_data(100)
test_DecisionTreeRegressor_depth(7, *data)

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


# 载入iris数据
def load_data():
    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target
    return cross_validation.train_test_split(X_train, y_train, test_size=0.25, random_state=0, stratify=y_train)
    # 其中的stratify参数表示采用的是分层抽样


# 分类决策树测试函数
def test_DecisionTreeClassifier(*data):
    X_train, X_test, y_train, y_test = data
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    print "Training score %f" % (clf.score(X_train, y_train))
    print "Testing score  %f" % (clf.score(X_test, y_test))


data = load_data()
test_DecisionTreeClassifier(*data)


# 观察不同的切分质量的评价准则的得分
def test_DecisionTreeClassifier_criterion(*data):
    X_train, X_test, y_train, y_test = data
    criterions = ['gini', 'entropy']
    for criterion in criterions:
        clf = DecisionTreeClassifier(criterion=criterion)
        clf.fit(X_train, y_train)
        print "Criterion:%s" % criterion
        print "Training score %f" % (clf.score(X_train, y_train))
        print "Testing score  %f" % (clf.score(X_test, y_test))

data = load_data()
test_DecisionTreeClassifier_criterion(*data)


# 探索不同深度的树对预测结果的影响
def test_DecisionTreeClassifier_depth(maxdepth,*data):
    X_train,X_test,y_train,y_test=data
    depths=np.arange(1,maxdepth)
    training_scores=[]
    testing_scores=[]
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth)
        clf.fit(X_train,y_train)
        training_scores.append(clf.score(X_train,y_train))
        testing_scores.append(clf.score(X_test,y_test))
    # plot
    fig = plt.figure()
    ax= fig.add_subplot(1,1,1)
    ax.plot(depths,training_scores,label='training score',marker='o')
    ax.plot(depths,testing_scores,label='testing score',marker='*')
    ax.set_xlabel("depth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Classification")
    ax.legend(loc="best",framealpha=0.5)
    plt.show()
data=load_data()
test_DecisionTreeClassifier_depth(8,*data)

