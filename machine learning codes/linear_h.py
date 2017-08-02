# -*- coding:utf-8 -*-
"""线性模型"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, discriminant_analysis, cross_validation

# 载入糖尿病人数据X_train, X_test, y_train, y_test = load_data()
X = np.vstack((X_train, X_test))
Y = np.vstack((y_train.reshape(y_train.size, 1), y_test.reshape(y_test.size, 1)))
lda = discriminant_analysis.LinearDiscriminantAnalysis()
lda.fit(X, Y)
converted_X = np.dot(X, np.transpose(lda.coef_)) + lda.intercept_
plot_LDA(converted_X, Y)


def load_data():  #  加载数据集的函数
    diabetes = datasets.load_diabetes()
    return cross_validation.train_test_split(diabetes.data, diabetes.target, \
                                             test_size=0.25, random_state=0)


# 返回一个元组
X_train, X_test, y_train, y_test = load_data()
X = np.vstack((X_train, X_test))
Y = np.vstack((y_train.reshape(y_train.size, 1), y_test.reshape(y_test.size, 1)))
lda = discriminant_analysis.LinearDiscriminantAnalysis()
lda.fit(X, Y)
converted_X = np.dot(X, np.transpose(lda.coef_)) + lda.intercept_
plot_LDA(converted_X, Y)


# 一般线性回归
def test_LinearRegression(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    print "Coefficents:%s,intercept:%.2f" % (regr.coef_, regr.intercept_)
    print "Resdiual mean of squares:%.2f" % np.mean((regr.predict(X_test) - y_test) ** 2)
    print "Score:%.2f" % regr.score(X_test, y_test)


data = load_data()
test_LinearRegression(*data)


# 岭回归
def test_ridge(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.Ridge()
    regr.fit(X_train, y_train)
    print "Coefficents:%s,intercept:%.2f" % (regr.coef_, regr.intercept_)
    print "Resdiual mean of squares:%.2f" % np.mean((regr.predict(X_test) - y_test) ** 2)
    print "Score:%.2f" % regr.score(X_test, y_test)


data = load_data()
test_ridge(*data)


# 不同的alpha值对预测性能的影响
def test_Ridge_alpha(*data):
    X_train, X_test, y_train, y_test = data
    alpha = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    scores = []
    for a in alpha:
        regr = linear_model.Ridge(alpha=a)
        regr.fit(X_train, y_train)
        scores.append(regr.score(X_test, y_test))
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alpha, scores)
    ax.set_xlabel("alpha")
    ax.set_ylabel("socre")
    ax.set_xscale('log')
    ax.set_title("Ridge Regression under different alpha values")
    plt.show()


data = load_data()
test_Ridge_alpha(*data)


# lasso回归
def test_lasso(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.Lasso()
    regr.fit(X_train, y_train)
    print "Coefficents:%s,intercept:%.2f" % (regr.coef_, regr.intercept_)
    print "Resdiual mean of squares:%.2f" % np.mean((regr.predict(X_test) - y_test) ** 2)
    print "Score:%.2f" % regr.score(X_test, y_test)


data = load_data()
test_lasso(*data)


# elasticnet回归
def test_ElasticNet(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.ElasticNet()
    regr.fit(X_train, y_train)
    print "Coefficents:%s,intercept:%.2f" % (regr.coef_, regr.intercept_)
    print "Resdiual mean of squares:%.2f" % np.mean((regr.predict(X_test) - y_test) ** 2)
    print "Score:%.2f" % regr.score(X_test, y_test)


data = load_data()
test_ElasticNet(*data)


# 不同的alpha值对预测性能的影响
def test_ElasticNet_alpha(*data):
    X_train, X_test, y_train, y_test = data
    alpha = np.logspace(-2, 2)
    rhos = np.linspace(0.01, 1)
    scores = []
    for a in alpha:
        for s in rhos:
            regr = linear_model.ElasticNet(alpha=a, l1_ratio=s)
            regr.fit(X_train, y_train)
            scores.append(regr.score(X_test, y_test))
    # plot
    alphas, rhos = np.meshgrid(alpha, rhos)
    scores = np.array(scores).reshape(alphas.shape)
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(alphas, rhos, scores, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("alpha")
    ax.set_ylabel("rho")
    ax.set_zlabel("score")
    ax.set_title("ElasticNet")
    plt.show()


data = load_data()
test_ElasticNet_alpha(*data)


# 载入鸢尾花数据
def load_data():
    iris = datasets.load_iris()
    return cross_validation.train_test_split(iris.data, iris.target, test_size=0.25, random_state=0,
                                             stratify=iris.target)


# 逻辑回归
def test_LogisticRegression(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.LogisticRegression()
    regr.fit(X_train, y_train)
    print "Coefficents:%s,intercept:%s" % (regr.coef_, regr.intercept_)
    print "Resdiual mean of squares:%.2f" % np.mean((regr.predict(X_test) - y_test) ** 2)
    print "Score:%.2f" % regr.score(X_test, y_test)


data = load_data()
test_LogisticRegression(*data)


# 逻辑回归的原生多类分类
def test_LogisticRegression_multinomial(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')
    regr.fit(X_train, y_train)
    print "Coefficents:%s,intercept:%s" % (regr.coef_, regr.intercept_)
    print "Resdiual mean of squares:%.2f" % np.mean((regr.predict(X_test) - y_test) ** 2)
    print "Score:%.2f" % regr.score(X_test, y_test)


data = load_data()
test_LogisticRegression_multinomial(*data)


# 考察正则化项的系数C对分类模型的预测性能的影响
def test_LogisticRegression_C(*data):
    X_train, X_test, y_train, y_test = data
    Cs = np.logspace(-2, 4, num=100)
    scores = []
    for C in Cs:
        regr = linear_model.LogisticRegression(C=C)
        regr.fit(X_train, y_train)
        scores.append(regr.score(X_test, y_test))
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(Cs, scores)
    ax.set_xlabel("C")
    ax.set_ylabel("score")
    ax.set_xscale('log')
    ax.set_title("LogisticRegression")
    plt.show()


data = load_data()
test_LogisticRegression_C(*data)


# LDA
def test_LinearDiscriminantAnalysis(*data):
    X_train, X_test, y_train, y_test = data
    lda = discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    print "Coefficents:%s,intercept:%s" % (lda.coef_, lda.intercept_)
    print "Score:%.2f" % lda.score(X_test, y_test)


data = load_data()
test_LinearDiscriminantAnalysis(*data)


# 原始数据经过LDA之后的数据集的情况,四维数据降维至三维
def plot_LDA(converted_X, y):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = 'rgb'
    markers = 'o*s'
    for target, color, marker in zip([0, 1, 2], colors, markers):
        pos = (y == target).ravel()   # 展平为一维度的ndarray
        X = converted_X[pos, :]
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], color=color, marker=marker, label="Label%d" % target)
    ax.legend(loc="best")
    fig.suptitle("Iris After LDA")
    plt.show()


X_train, X_test, y_train, y_test = load_data()
X = np.vstack((X_train, X_test))
Y = np.vstack((y_train.reshape(y_train.size, 1), y_test.reshape(y_test.size, 1)))
lda = discriminant_analysis.LinearDiscriminantAnalysis()
lda.fit(X, Y)
converted_X = np.dot(X, np.transpose(lda.coef_)) + lda.intercept_
plot_LDA(converted_X, Y)


# 考察不同的solver对预测性能的影响
def test_LinearDIscriminantAnalysis_solver(*data):
    X_train,X_test,y_train,y_test=data
    solvers=['svd','lsqr','eigen']
    for solver in solvers:
        lda = discriminant_analysis.LinearDiscriminantAnalysis(solver=solver)
        lda.fit(X_train,y_train)
        print "Score at solver=%s:%.2f" %(solver,lda.score(X_test,y_test))

data = load_data()
test_LinearDIscriminantAnalysis_solver(*data)