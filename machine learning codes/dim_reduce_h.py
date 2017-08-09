# -*- coding:utf-8 -*-
"""dimension reduction"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition, manifold


# 载入数据集
def load_data():
    iris = datasets.load_iris()
    return iris.data, iris.target


# 测试函数
def test_PCA(*data):
    X, y = data
    pca = decomposition.PCA(n_components=None)
    pca.fit(X)
    print "explained variance ratio:%s" % str(pca.explained_variance_ratio_)


data = load_data()
test_PCA(*data)


# 绘制降维后的样本分布图
def plot_PCA(*data):
    X, y = data
    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    X_r = pca.transform(X)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    for label, color in zip(np.unique(y), colors):
        ax.scatter(X_r[y == label, 0], X_r[y == label, 1], label="target=%d" % label, color=color)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("Y[0]")
    ax.legend(loc="best")
    ax.set_title("PCA")
    plt.show()


data = load_data()
plot_PCA(*data)
