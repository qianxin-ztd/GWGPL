#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Created on 2024/1/17 14:36
@Author  : Qian xin
@File    : Compared_method.py
"""

from util import *
import itertools

class one_point_estimation():
    def __init__(self, add_weight=False,
                 lamd=1e-3, tol=1e-3, max_iter=300, threshold=0.001):
        """
        Compared_method
        :param add_weight:
        :param lamd:
        :param tol:
        :param max_iter:
        """
        self.lamd = lamd
        self.tol = tol
        self.max_iter = max_iter
        self.add_weight = add_weight
        self.threshold = threshold

    def Lasso(self, X, y):
        b = 0
        rss = lambda X, y, w, b: np.linalg.norm(y - X * w - b, ord=2) ** 2 + self.lamd * np.linalg.norm(w, ord=1)
        # 初始化回归系数w.
        n, p = X.shape
        w = np.matrix(np.zeros((p, 1)))
        r = rss(X, y, w, b)
        # 使用坐标下降法优化回归系数w
        niter = itertools.count(1)
        for it in niter:
            for k in range(p):  # 对于每一个特征
                # 计算常量值z_k和p_k
                z_k = np.linalg.norm(X[:, k], ord=2) ** 2
                p_k = 0
                for i in range(n):
                    p_k += X[i, k] * (y[i, 0] - sum([X[i, j] * w[j, 0] for j in range(p) if j != k]) - b)
                p_k = p_k
                if p_k < -self.lamd / 2:
                    w_k = (p_k + self.lamd / 2) / z_k
                elif p_k > self.lamd / 2:
                    w_k = (p_k - self.lamd / 2) / z_k
                else:
                    w_k = 0
                w[k, 0] = w_k
            b = np.sum(y - X @ w) / n
            r_prime = rss(X, y, w, b)
            delta = abs(r_prime - r)
            r = r_prime
            # print('Iteration: {}, delta = {}'.format(it, delta))
            if delta < self.tol or it > self.max_iter:
                # print("Converged. itr={}".format(it))
                break
        self.coef_ = np.array(w).reshape(1, -1)  # 主要的回归系数
        for i in range(self.coef_.shape[1]):
            if abs(self.coef_[:, i]) < self.threshold:
                self.coef_[:, i] = 0
            else:
                continue
        self.intercept_ = b  # 截距项
        return self

    def GWL(self, X, X_b, y, aer):
        rss = lambda X, y, X_b, w, b: (y - X * w - X_b * b).T * (y - X * w - X_b * b) + self.lamd * np.linalg.norm(w, ord=1)
        # 初始化回归系数w
        n, p = X.shape
        w = np.matrix(np.zeros((p, 1)))
        b = 0
        r = rss(X, y, X_b, w, b)
        # 使用坐标下降法优化回归系数w
        niter = itertools.count(1)
        for it in niter:
            for k in range(p):
                # 计算常量值z_k和p_k
                z_k = np.linalg.norm(X[:, k], ord=2) ** 2
                p_k = 0
                for i in range(n):
                    p_k += X[i, k] * (
                            y[i, 0] - sum([X[i, j] * w[j, 0] for j in range(p) if j != k]) - X_b[i, 0] * b)
                p_k = p_k
                if p_k < -self.lamd / 2:
                    w_k = (p_k + self.lamd / 2) / z_k
                elif p_k > self.lamd / 2:
                    w_k = (p_k - self.lamd / 2) / z_k
                else:
                    w_k = 0
                w[k, 0] = w_k
            b = np.sum(np.multiply(aer, y) - np.multiply(aer, X) @ w) / np.sum(np.power(aer, 2))
            r_prime = rss(X, y, X_b, w, b)
            delta = abs(r_prime - r)
            r = r_prime
            # if self.verbose and it % self.verbose_interval == 0:
            # print('Iteration: {}, delta = {}'.format(it, delta))
            if delta < self.tol or it > self.max_iter:
                # print("Converged. itr={}".format(it))
                break
        self.coef_ = np.array(w).reshape(1, -1)  # 主要的回归系数
        for i in range(self.coef_.shape[1]):
            if abs(self.coef_[:, i]) < self.threshold:
                self.coef_[:, i] = 0
            else:
                continue
        self.intercept_ = b  # 截距项
        return self


    def fit(self, X, y, g_coords, loc, aerfa):
        if isinstance(X, pd.DataFrame): X = X.values
        if isinstance(y, pd.Series): y = y.values

        if not self.add_weight and self.lamd == 0:  # LSQ
            X = add_intercept(X)
            beta = Least_squares(X, y, g_coords, loc)
            self.coef_ = beta[:-1, :].reshape(1, -1)
            self.intercept_ = beta[-1, :]
            return self
        if self.add_weight and self.lamd == 0:
            X, w = add_intercept(X), []
            for i in range(len(X)):
                for j in range(len(aerfa)):
                    if X[i][0] == aerfa.index[j][0] and X[i][1] == aerfa.index[j][1]:
                        w.append(aerfa[aerfa.index[j]])
            beta = localWeightRegression(X, y, w)
            self.coef_ = beta[:-1, :].reshape(1, -1)
            self.intercept_ = beta[-1, :]
            return self
        if not self.add_weight and self.lamd:
            XX, yy = X[list((X[:, 0] == g_coords[loc][0]) & (X[:, 1] == g_coords[loc][1])), :][:, 2:], \
                     y[list((X[:, 0] == g_coords[loc][0]) & (X[:, 1] == g_coords[loc][1]))]
            X, y = np.matrix(XX), np.matrix(yy)  # 其中并不含截距项，单独考虑
            y = y.reshape(-1, 1)
            self.Lasso(X, y)
        if self.add_weight and self.lamd != 0:
            w = []  # 得到和y相对应的在该中心点下的地理权重
            for i in range(len(X)):
                for j in range(len(aerfa)):
                    if X[i][0] == aerfa.index[j][0] and X[i][1] == aerfa.index[j][1]:
                        w.append(aerfa[aerfa.index[j]])
            aer = np.matrix(w).reshape(-1, 1)  # 得到和y相对应的在该中心点下的地理权重
            aer = np.power(aer, 0.5)
            X, y, X_b = np.matrix(X[:, 2:]), np.matrix(y), np.ones(X.shape[0]).reshape(-1, 1)
            y = y.reshape(-1, 1)
            X, y, X_b = np.multiply(aer, X), np.multiply(aer, y), np.multiply(aer, X_b)
            self.GWL(X, X_b, y, aer)


    def predict(self, X):   #其中X为X_test，y为y_test
        W = np.hstack((self.coef_, self.intercept_.reshape(-1, 1)))
        return X @ W.reshape(-1, 1)