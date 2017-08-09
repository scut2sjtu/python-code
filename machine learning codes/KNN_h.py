# -*- coding:utf-8 -*-
"""KNN"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets, cross_validation


# 加载数据集
def load_classification_data():
    digits = datasets.load_digits()
    return cross_validation.train_test_split(digits.data, digits.target, test_size=0.25, random_state=0,
                                             stratify=digits.target)


def create_regression_data(n):
    X = np.random.rand(n, 1)
    y = np.sin(X).ravel()
    y[::5] += 1 * (0.5 - np.random.rand(int(n / 5)))
    return cross_validation.train_test_split(X, y, test_size=0.25, random_state=0)


# 测试函数
def test_KNeighborsClassifier(*data):
    X_train, X_test, y_train, y_test = data
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)
    print "Training Score:%f" % clf.score(X_train, y_train)
    print "Testing Score:%f" % clf.score(X_test, y_test)


data = load_classification_data()
test_KNeighborsClassifier(*data)


# 考察k值以及投票策略对于预测性能的影响
def test_KNeighborsClassifier_k_w(*data):
    X_train, X_test, y_train, y_test = data
    Ks = np.linspace(1, y_train.size, num=100, endpoint=False, dtype='int')
    weights = ['uniform', 'distance']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for weight in weights:
        training_score = []
        testing_score = []
        for k in Ks:
            clf = neighbors.KNeighborsClassifier(weights=weight, n_neighbors=k)
            clf.fit(X_train, y_train)
            training_score.append(clf.score(X_train, y_train))
            testing_score.append(clf.score(X_test, y_test))
        ax.plot(Ks, training_score, label="training score:weight=%s" % weight)
        ax.plot(Ks, testing_score, label="tesing score:weight=%s" % weight)
    ax.legend(loc='best')
    ax.set_xlabel("K")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("KNeighborsClassifier")
    plt.show()


data = load_classification_data()
test_KNeighborsClassifier_k_w(*data)


# KNN回归测试函数
def test_KNeighborsRegressor(*data):
    X_train, X_test, y_train, y_test = data
    regr = neighbors.KNeighborsRegressor()
    regr.fit(X_train, y_train)
    print "Training Score:%f" % regr.score(X_train, y_train)
    print "Testing Score:%f" % regr.score(X_test, y_test)


X_train, X_test, y_train, y_test = create_regression_data(1000)
test_KNeighborsRegressor(X_train, X_test, y_train, y_test)
