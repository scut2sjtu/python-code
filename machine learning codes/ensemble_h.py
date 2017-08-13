# -*- coding:utf-8 -*-
"""ensemble learning""""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, cross_validation, ensemble


# 加载回归数据集
def load_data_regression():
    diabetes = datasets.load_diabetes()
    return cross_validation.train_test_split(diabetes.data, diabetes.target, test_size=0.25, random_state=0)


# 加载分类数据集
def load_data_classification():
    digits = datasets.load_digits()
    return cross_validation.train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)


# 使用AdaBoostClassifier的函数
def test_AdaBoostClassifier(*data):
    X_train, X_test, y_train, y_test = data
    clf = ensemble.AdaBoostClassifier(learning_rate=0.1)
    clf.fit(X_train, y_train)
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    estimators_num = len(clf.estimators_)
    X = range(1, estimators_num + 1)
    ax.plot(X, list(clf.staged_score(X_train, y_train)), label="Training score")
    ax.plot(X, list(clf.staged_score(X_test, y_test)), label="Testing score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="best")
    ax.set_title("AdaBoostClassifier")
    plt.show()


data = load_data_classification()
test_AdaBoostClassifier(*data)


# AdaBoostRegressor的测试函数
def test_AdaBoostRegressor(*data):
    X_train, X_test, y_train, y_test = data
    regr = ensemble.AdaBoostRegressor()
    regr.fit(X_train, y_train)
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    estimators_num = len(regr.estimators_)
    X = range(1, estimators_num + 1)
    ax.plot(X, list(regr.staged_score(X_train, y_train)), label="Training Score")
    ax.plot(X, list(regr.staged_score(X_test, y_test)), label="Testing Score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="best")
    ax.set_title("AdaBoostRegressor")
    plt.show()


data = load_data_regression()
test_AdaBoostRegressor(*data)


# GradientBoostingClassifier函数
def test_GradientBoostingClassifier(*data):
    X_train, X_test, y_train, y_test = data
    clf = ensemble.GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    print "Training Score:%f" % clf.score(X_train, y_train)
    print "Testing Score:%f" % clf.score(X_test, y_test)


data = load_data_classification()
test_GradientBoostingClassifier(*data)


# GradientBoostingRegressor函数
def test_GradientBoostingRegressor(*data):
    X_train,X_test,y_train,y_test=data
    regr=ensemble.GradientBoostingRegressor()
    regr.fit(X_train,y_train)
    print "Training Score:%f"%regr.score(X_train,y_train)
    print "Testing Score:%f"%regr.score(X_test,y_test)
data = load_data_regression()
test_GradientBoostingRegressor(*data)


# RandomForestClassifier测试函数
def test_RandomForestClassifier(*data):
    X_train,X_test,y_train,y_test=data
    clf=ensemble.RandomForestClassifier()
    clf.fit(X_train,y_train)
    print "Training Score:%f"%clf.score(X_train,y_train)
    print "TEsting Score:%f"%clf.score(X_test,y_test)
data = load_data_classification()
test_RandomForestClassifier(*data)

# RandomForestRegressor测试函数
def test_RandomForestRegressor(*data):
    X_train,X_test,y_train,y_test=data
    regr=ensemble.RandomForestRegressor()
    regr.fit(X_train,y_train)
    print "Training Score:%f"% regr.score(X_train,y_train)
    print "Testing Score:%f"% regr.score(X_test,y_test)
data = load_data_regression()
test_RandomForestRegressor(*data)

