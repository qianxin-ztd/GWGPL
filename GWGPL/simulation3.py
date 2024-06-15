
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2024/1/17 14:33
@Author  : Qian xin
@File    : simulation.py
"""
import numpy as np
import pandas as pd

from Compared_method import one_point_estimation
from sklearn import metrics
import random
from sklearn.model_selection import StratifiedShuffleSplit
from util import *
RANDOM_STATE = 42
from GWGPL import GWGPL
import os
import matplotlib.pyplot as plt

def generate_one_point_data(N, beta, setting, rho_1, rho_2, sigma):   # 生成一个地点的数据
    # setting为0表示全部为离散变量，setting为1表示一半离散变量，一半连续变量，setting为2表示全部为连续变量
    if setting == 'discrete':
        # 多元正态数据，第一部分为非零系数部分，第二部分为零系数部分
        Sigma_11, Sigma_22 = np.full((p1 + p2, p1 + p2), rho_1), np.full((p_noise, p_noise), rho_1)
        Sigma_12, Sigma_21 = np.full((p1 + p2, p_noise), rho_2), np.full((p_noise, p1 + p2), rho_2)
        Sigma = np.vstack((np.hstack((Sigma_11, Sigma_12)), np.hstack((Sigma_21, Sigma_22))))
        np.fill_diagonal(Sigma, 1)  # 因为每个变量自身的方差为1，所以两个变量间的协方差等于相关系数
        data = np.random.multivariate_normal(np.zeros(p1 + p2 + p_noise), Sigma, N)

        data[:, p1:p1 + p2//2] = np.digitize(data[:, p1:p1 + p2], [np.quantile(data[:, p1:p1 + p2], 0.5)])
        data[:, p1 + p2//2:p1 + p2] = np.digitize(data[:, p1:p1 + p2], [np.quantile(data[:, p1:p1 + p2], 0.25)])

        y = np.c_[data[:, :p1 + p2], np.ones(len(data[:, :p1 + p2]))] @ beta + np.random.multivariate_normal(
            np.zeros(N), np.identity(N), N)[0,] * sigma
        dataset = np.concatenate((data.astype(np.float32), y.astype(np.float32).reshape(-1, 1)), axis=1)
        dataset = pd.DataFrame(dataset, columns=[*[f'x{i + 1}' for i in range(data.shape[1])], 'y'])

    elif setting == 'half':
        # 多元正态数据，第一部分为非零系数部分，第二部分为零系数部分
        Sigma_11, Sigma_22 = np.full((p1+p2, p1+p2), rho_1), np.full((p_noise, p_noise), rho_1)
        Sigma_12, Sigma_21 = np.full((p1+p2, p_noise), rho_2), np.full((p_noise, p1 + p2), rho_2)
        Sigma = np.vstack((np.hstack((Sigma_11, Sigma_12)), np.hstack((Sigma_21, Sigma_22))))
        np.fill_diagonal(Sigma, 1)  # 因为每个变量自身的方差为1，所以两个变量间的协方差等于相关系数
        data = np.random.multivariate_normal(np.zeros(p1+p2+p_noise), Sigma, N)

        data[:, p1:p1+p2] = np.digitize(data[:, p1:p1+p2], [np.quantile(data[:, p1:p1+p2], 0.5)])

        y = np.c_[data[:, :p1+p2], np.ones(len(data[:, :p1+p2]))] @ beta + np.random.multivariate_normal(
            np.zeros(N), np.identity(N), N)[0,] * sigma
        dataset = np.concatenate((data.astype(np.float32), y.astype(np.float32).reshape(-1, 1)), axis=1)
        dataset = pd.DataFrame(dataset, columns=[*[f'x{i + 1}' for i in range(data.shape[1])], 'y'])

    elif setting == 'continue':
        # 多元正态数据，第一部分为非零系数部分，第二部分为零系数部分
        Sigma_11, Sigma_22 = np.full((p1 + p2, p1 + p2), rho_1), np.full((p_noise, p_noise), rho_1)
        Sigma_12, Sigma_21 = np.full((p1 + p2, p_noise), rho_2), np.full((p_noise, p1 + p2), rho_2)
        Sigma = np.vstack((np.hstack((Sigma_11, Sigma_12)), np.hstack((Sigma_21, Sigma_22))))
        np.fill_diagonal(Sigma, 1)  # 因为每个变量自身的方差为1，所以两个变量间的协方差等于相关系数
        data = np.random.multivariate_normal(np.zeros(p1 + p2 + p_noise), Sigma, N)
        y = np.c_[data[:, :p1 + p2], np.ones(len(data[:, :p1 + p2]))] @ beta + np.random.multivariate_normal(
            np.zeros(N), np.identity(N), N)[0,:] * sigma
        dataset = np.concatenate((data.astype(np.float32), y.astype(np.float32).reshape(-1, 1)), axis=1)
        dataset = pd.DataFrame(dataset, columns=[*[f'x{i + 1}' for i in range(data.shape[1])], 'y'])
    return dataset

def generate_repeat_data(repeat_number, setting, rho_1, rho_2, sigma):
    beta = np.full((l, p1 + p2 + 1), np.nan)  # 初始化回归系数
    a = np.linspace(1.5, 4, p1 + p2 + 1)  # 得到不同的特征
    for feature in range(p1 + p2 + 1):  # 得到所有地点的回归系数
        beta[:, feature] = (np.sqrt(0.5 - (np.sqrt((g_coords[:, 0] - 0.5) ** 2 + (g_coords[:, 1] - 0.5) ** 2)) ** 2)) / a[feature]  # 得到beta值
        # beta[:, feature] = (np.sqrt(1 - (np.sqrt((g_coords[:, 0] - 1) ** 2 + (g_coords[:, 1]) ** 2)) ** 2)) / a[feature]  # 得到beta值
    threshold = np.min(beta[:, -2]) / 10000
    repeat_data = []
    repeat_number = repeat_number
    for aa in range(repeat_number):
        np.random.seed(1314 + aa)
        simdat = []
        for i in range(l):  # 对每一个地点
            datset = generate_one_point_data(n[i], beta[i, :], setting, rho_1, rho_2, sigma)
            simdat.append(datset)  # 每个location存一个，即simdat会有p个元素
        data_with_loc = pd.DataFrame(
            columns=['location', *[i for i in simdat[0].columns if i not in ('location', 'y')], 'y'])
        for i in range(l):
            simdat[i]['location'] = i + 1
            simdat[i] = simdat[i][['location', *[i for i in simdat[i].columns if i not in ('location', 'y')], 'y']]
            data_with_loc = pd.concat([data_with_loc, simdat[i]], axis=0, ignore_index=True)
        data_generate = pd.DataFrame(columns=['loc_x', 'loc_y', *[i for i in simdat[i].columns if i not in ('location')]])
        for loc in range(l):
            data = data_with_loc[data_with_loc['location'] == loc + 1]
            data = data.drop('location', axis=1)
            data.insert(0, 'loc_x', g_coords[loc][0])
            data.insert(1, 'loc_y', g_coords[loc][1])
            data_generate = pd.concat([data_generate, data], axis=0, ignore_index=True)
        repeat_data.append(data_generate)
    return repeat_data, beta, threshold


if __name__ == '__main__':
    ############ data generating ##############
    setting = 'half'
    p1, p2, p_noise = 5, 5, 10  # number of continuous covariates, discrete covariates and noise covariates
    p, l, sample_average = p1 + p2 + p_noise, 10, 30  # number of covariates and number of locations
    # np.random.seed(0)
    # n = np.random.normal(sample_average, sample_average / 4, l).astype(int)  # 生成的样本不同样本取值不同
    n = [sample_average for i in range(l)]
    print('setting=' + setting + ',  p1, p2, p_noise=' + str(p1) +','+ str(p2) +','+ str(p_noise) + ',  n=' + str(sample_average))
    # g_coords = np.array([
    #                      [0.08375981, 0.02825333],
    #                      [0.33665332, 0.01964299],
    #                      [0.99523272, 0.78500807],
    #                      [0.38494575, 0.25065348],
    #                      [0.94082562, 0.93311949],
    #                      [0.90742148, 0.66599465],
    #                      [0.93718995, 0.85749397],
    #                      [0.92127660, 0.34535785],
    #                      [0.73589767, 0.06708488],
    #                      [0.52188345, 0.04101815]])

    g_coords = np.array([
        [0.44201802, 0.56235160],
        [0.40734233, 0.57933696],
        [0.60023565, 0.565987510],
        [0.53526954, 0.523568520],
        [0.43021354, 0.415462460],
        [0.45239755, 0.512314650],
        [0.55423135, 0.454313250],
        [0.50232135, 0.422531560],
        [0.46542356, 0.542351540],
        [0.58456213, 0.421536970]])

    # plt.scatter(g_coords[:,0], g_coords[:,1])
    # plt.show()
    # data_generate = generate_data()   # 进行单次单地点实验
    repeat_number = 10
    RHO_1, rho_2, SIGMA = [0, 0.5], 0, [1/3, 1/2, 1]
    for sigma in SIGMA:
        for rho_1 in RHO_1:
            repeat_data, beta, threshold = generate_repeat_data(repeat_number, setting, rho_1, rho_2, sigma)  # 重复进行多次单地点实验，其中beta值保持不变，让X和误差项的值发生改变，从而改变y值
            loss = np.zeros([5, repeat_number])  # 用来存贮每种方法在重复实验中的误差
            coef_loss = np.zeros([5, repeat_number])  # 用来存贮每种方法在重复实验中的系数估计误差
            coef_selection_n1, coef_selection_n2, coef_selection_n3, coef_selection_n4= np.zeros([3, repeat_number]), np.zeros([3, repeat_number]), \
                np.zeros([3, repeat_number]), np.zeros([3, repeat_number])  # 用来存贮lasso方法的变量选择情况
            b_min, b_max = bw_selection(g_coords)
            bw = np.linspace(b_min + 0.01, b_max * 2, 10)  # 提前给定的bw的取值
            lamda = np.linspace(0.1, 10, 30)
            for repeat in range(repeat_number):
                try:
                    loss_par_lamda = np.zeros([5, len(lamda)])
                    loss_par_bw = np.zeros([5, len(bw)])
                    #########################导入数据并进行数据划分###############################
                    data_generate = repeat_data[repeat]
                    split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=RANDOM_STATE)
                    split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_STATE)
                    for train_index, test_index in split1.split(data_generate, data_generate["loc_x"]):
                        train_o, test_o = data_generate.iloc[train_index, :], data_generate.iloc[test_index, :]
                        X_train_o, y_train_o = train_o.iloc[:, 0:p + 2], train_o.iloc[:, p + 2]
                        X_test_o, y_test_o = test_o.iloc[:, 0:p + 2], test_o.iloc[:, p + 2]
                    for train_index, test_index in split2.split(train_o, train_o["loc_x"]):
                        train_c, test_c = train_o.iloc[train_index, :], train_o.iloc[test_index, :]
                        X_train_x, y_train_x = train_c.iloc[:, 0:p + 2], train_c.iloc[:, p + 2]
                        X_test_c, y_test_c = test_c.iloc[:, 0:p + 2], test_c.iloc[:, p + 2]
                    X_train_x, X_test_c, X_test_o, y_train_x, y_test_c, y_test_o = np.array(X_train_x), np.array(X_test_c), \
                        np.array(X_test_o), np.array(y_train_x), np.array(y_test_c), np.array(y_test_o)
                    ###################################计算基本信息###################################
                    for i in range(len(lamda)):
                        lamd, gwr_bw = lamda[i], b_max / 2
                        print("lamd=", lamd, "bw=", gwr_bw, ",calcaulating...")
                        Aerfa = Kernel(g_coords, gwr_bw, 'bi-square')  # 空间矩阵计算
                        lsq_loss, weight_loss, lsq_lasso_loss, weight_lasso_loss, sim_loss = [], [], [], [], []  # 用来存储每个方法在每个地点的误差
                        simultaneous_model = GWGPL(lamd=lamd, tol=1e-3, max_iter=1000, threshold=threshold)
                        simultaneous_model.fit(X_train_x, y_train_x, Aerfa, g_coords, p, l)
                        for loc in range(len(g_coords)):
                            aerfa = Aerfa.iloc[:, loc]
                            XX_test, yy_test = add_intercept(X_test_c[list((X_test_c[:, 0] == g_coords[loc][0])&(X_test_c[:, 1] == g_coords[loc][1])), :][:, 2:]), \
                                y_test_c[list((X_test_c[:, 0] == g_coords[loc][0]) & (X_test_c[:, 1] == g_coords[loc][1]))]  # 测试的时候使用该地点的数据进行测试
                            if len(XX_test) == 0:  # 如果该地点没有数据就不再进行预测
                                continue
                            #########################进行单地点最小二乘回归###################################
                            lsq_model = one_point_estimation(add_weight=False, lamd=0, tol=1e-10, max_iter=1000, threshold=threshold)
                            lsq_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
                            y_lsq = lsq_model.predict(XX_test)
                            lsq_loss.append(metrics.mean_squared_error(yy_test, y_lsq) ** 0.5)
                            ##########################进行局部加权回归##################################
                            weight_model = one_point_estimation(add_weight=True,  lamd=0, tol=1e-10, max_iter=1000, threshold=threshold)
                            weight_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
                            y_weight = weight_model.predict(XX_test)
                            weight_loss.append(metrics.mean_squared_error(yy_test, y_weight) ** 0.5)
                            #########################进行最小二乘+Lasso##################################
                            lasso_model = one_point_estimation(add_weight=False, lamd=lamd, tol=1e-10, max_iter=1000, threshold=threshold)
                            lasso_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
                            y_lsq_lasso = lasso_model.predict(XX_test)
                            lsq_lasso_loss.append(metrics.mean_squared_error(yy_test, np.array(y_lsq_lasso)) ** 0.5)
                            #######################进行局部加权+Lasso##################################
                            weight_lasso_model = one_point_estimation(add_weight=True, lamd=lamd, tol=1e-10, max_iter=1000, threshold=threshold)
                            weight_lasso_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
                            y_weight_lasso = weight_lasso_model.predict(XX_test)
                            weight_lasso_loss.append(metrics.mean_squared_error(yy_test, y_weight_lasso) ** 0.5)
                            # # ##############################进行整体同时估计################################
                            y_sim = simultaneous_model.predict(XX_test, loc, p)
                            sim_loss.append(metrics.mean_squared_error(yy_test, np.array(y_sim)) ** 0.5)
                        loss_par_lamda[0, i] = np.average(lsq_loss) if len(lsq_loss) != 0 else 0
                        # loss_par_lamda[1, i] = np.average(weight_loss) if len(weight_loss) != 0 else 0
                        loss_par_lamda[2, i] = np.average(lsq_lasso_loss) if len(lsq_lasso_loss) != 0 else 0
                        loss_par_lamda[3, i] = np.average(weight_lasso_loss) if len(weight_lasso_loss) != 0 else 0
                        loss_par_lamda[4, i] = np.average(sim_loss) if len(sim_loss) != 0 else 0

                    for i in range(len(bw)):
                        lamd1, lamd2 = lamda[np.argmin(loss_par_lamda[3, :])], lamda[np.argmin(loss_par_lamda[4, :])]
                        gwr_bw = bw[i]
                        print("lamd1=", lamd1, "lamd2=", lamd2, "bw=", gwr_bw, ",calcaulating...")
                        Aerfa = Kernel(g_coords, gwr_bw, 'bi-square')  # 空间矩阵计算
                        ###################################计算基本信息###################################
                        lsq_loss, weight_loss, lsq_lasso_loss, weight_lasso_loss, sim_loss = [], [], [], [], []  # 用来存储每个方法在每个地点的误差
                        simultaneous_model = GWGPL(lamd=lamd2, tol=1e-10, max_iter=1000, threshold=threshold)
                        simultaneous_model.fit(X_train_x, y_train_x, Aerfa, g_coords, p, l)
                        for loc in range(len(g_coords)):
                            aerfa = Aerfa.iloc[:, loc]
                            XX_test, yy_test = add_intercept(X_test_c[list((X_test_c[:, 0] == g_coords[loc][0]) & (X_test_c[:, 1] == g_coords[loc][1])), :][:, 2:]), \
                                               y_test_c[list((X_test_c[:, 0] == g_coords[loc][0]) & (X_test_c[:, 1] == g_coords[loc][1]))]  # 测试的时候使用该地点的数据进行测试
                            if len(XX_test) == 0:  # 如果该地点没有数据就不再进行预测
                                continue
                            #########################进行单地点最小二乘回归###################################
                            # lsq_model = one_point_estimation(lsq=True, add_weight=False,
                            #                                  # lsq为True的时候进行最小二乘, add_weight为True的时候进行局部加权
                            #                                  lamd=0, tol=0.001, max_iter=100)  # lamd为0时进行局部加权回归
                            # lsq_model.fit(X_train, y_train)
                            # y_lsq = lsq_model.predict(XX_test)
                            # lsq_loss.append(sum(abs(yy_test - y_lsq.reshape(1, -1))[0]))
                            ##########################进行局部加权回归##################################
                            weight_model = one_point_estimation(add_weight=True,lamd=0, tol=1e-10, max_iter=1000, threshold=threshold)
                            weight_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
                            y_weight = weight_model.predict(XX_test)
                            weight_loss.append(metrics.mean_squared_error(yy_test, y_weight) ** 0.5)
                            # # ########################进行最小二乘+Lasso##################################
                            # lsq_lasso_model = one_point_estimation(lsq=True, add_weight=False,  # lsq为True的时候进行最小二乘
                            #                                        lamd=lamd, tol=0.001, max_iter=100)  # lamd为0时进行局部加权回归
                            # lsq_lasso_model.fit(X_train, y_train)
                            # y_lsq_lasso = lsq_lasso_model.predict(XX_test)
                            # lsq_lasso_loss.append(sum(abs(yy_test - np.array(y_lsq_lasso).reshape(1, -1))[0]))
                            #######################进行局部加权+Lasso##################################
                            weight_lasso_model = one_point_estimation(add_weight=True, lamd=lamd1, tol=1e-10, max_iter=1000, threshold=threshold)  # lamd为0时进行局部加权回归
                            weight_lasso_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
                            y_weight_lasso = weight_lasso_model.predict(XX_test)
                            weight_lasso_loss.append(metrics.mean_squared_error(yy_test, y_weight_lasso) ** 0.5)
                            # # ##############################进行整体同时估计################################
                            y_sim = simultaneous_model.predict(XX_test, loc, p)
                            sim_loss.append(metrics.mean_squared_error(yy_test, np.array(y_sim)) ** 0.5)
                        # loss_par_bw[0, i] = np.average(lsq_loss) if len(lsq_loss) != 0 else 0
                        loss_par_bw[1, i] = np.average(weight_loss) if len(weight_loss) != 0 else 0
                        # loss_par_bw[2, i] = np.average(lsq_lasso_loss) if len(lsq_lasso_loss) != 0 else 0
                        loss_par_bw[3, i] = np.average(weight_lasso_loss) if len(weight_lasso_loss) != 0 else 0
                        loss_par_bw[4, i] = np.average(sim_loss) if len(sim_loss) != 0 else 0
                    print("################本次模拟中选出的最佳参数：##################")
                    print("Weight_model (The best bw):", bw[np.argmin(loss_par_bw[1, :])])
                    print("Lasso_model (The best lamda):", lamda[np.argmin(loss_par_lamda[2, :])])
                    print("Weight_Lasso_model (The best lamda):", lamda[np.argmin(loss_par_lamda[3, :])], "(The best bw):", bw[np.argmin(loss_par_bw[3, :])])
                    print("simultaneous_model (The best lamda):", lamda[np.argmin(loss_par_lamda[4, :])], "(The best bw):", bw[np.argmin(loss_par_bw[4, :])])

                    print('###############开始进行测试集测试########################')
                    lsq_loss, weight_loss, lasso_loss, weight_lasso_loss, sim_loss = [], [], [], [], []  # 用来存储每个方法在每个地点的预测误差
                    lsq_coef_loss, weight_coef_loss, lasso_coef_loss, weight_lasso_coef_loss, sim_coef_loss = [], [], [], [], []
                    lsq_coef, weight_coef, lasso_coef, weight_lasso_coef = [], [], [], []
                    lsq_intercept, weight_intercept, lasso_intercept, weight_lasso_intercept = [], [], [], []
                    lasso_tn, weight_lasso_tn, sim_tn = 0, 0, 0
                    lasso_fp, weight_lasso_fp, sim_fp = 0, 0, 0
                    lasso_fn, weight_lasso_fn, sim_fn = 0, 0, 0
                    lasso_tp, weight_lasso_tp, sim_tp = 0, 0, 0
                    ########################GWGPL#############################
                    gwr_bw = bw[np.argmin(loss_par_bw[4, :])]
                    Aerfa = Kernel(g_coords, gwr_bw, 'bi-square')  # 空间矩阵计算
                    simultaneous_model = GWGPL(lamd=lamda[np.argmin(loss_par_lamda[4, :])], tol=1e-10, max_iter=1000, threshold=threshold)
                    simultaneous_model.fit(X_train_x, y_train_x, Aerfa, g_coords, p, l)
                    for lst in simultaneous_model.coef_.reshape((l, p))[:, p1 + p2:].T.tolist():
                        for a in lst:
                            if abs(a) < 1e-6:
                                sim_tn += 1
                    for lst in simultaneous_model.coef_.reshape((l, p))[:, p1 + p2:].T.tolist():
                        for a in lst:
                            if abs(a) > 1e-6:
                                sim_fp += 1
                    for lst in simultaneous_model.coef_.reshape((l, p))[:, :p1 + p2].T.tolist():
                        for a in lst:
                            if abs(a) < 1e-6:
                                sim_fn += 1
                    for lst in simultaneous_model.coef_.reshape((l, p))[:, :p1 + p2].T.tolist():
                        for a in lst:
                            if abs(a) > 1e-6:
                                sim_tp += 1
                    for loc in range(len(g_coords)):
                        XX_test, yy_test = add_intercept(X_test_o[list((X_test_o[:, 0] == g_coords[loc][0]) & (X_test_o[:, 1] == g_coords[loc][1])), :][:,2:]), \
                                                         y_test_o[list((X_test_o[:, 0] == g_coords[loc][0]) & (X_test_o[:, 1] == g_coords[loc][1]))]  # 测试的时候使用该地点的数据进行测试
                        if len(XX_test) == 0:  # 如果该地点没有数据就不再进行预测
                            continue
                        # ##############################进行整体同时估计################################
                        y_sim = simultaneous_model.predict(XX_test, loc, p)
                        sim_loss.append(metrics.mean_squared_error(yy_test, np.array(y_sim)) ** 0.5)

                    ########################lasso+GWR###############################
                    gwr_bw = bw[np.argmin(loss_par_bw[3, :])]
                    Aerfa = Kernel(g_coords, gwr_bw, 'bi-square')  # 空间矩阵计算
                    for loc in range(len(g_coords)):
                        aerfa = Aerfa.iloc[:, loc]
                        XX_test, yy_test = add_intercept(X_test_o[list((X_test_o[:, 0] == g_coords[loc][0]) & (X_test_o[:, 1] == g_coords[loc][1])), :][:, 2:]), \
                                                         y_test_o[list((X_test_o[:, 0] == g_coords[loc][0]) & (X_test_o[:, 1] == g_coords[loc][1]))] # 测试的时候使用该地点的数据进行测试
                        if len(XX_test) == 0:  # 如果该地点没有数据就不再进行预测
                            continue
                        ########################进行局部加权+Lasso##################################
                        weight_lasso_model = one_point_estimation(add_weight=True, lamd=lamda[np.argmin(loss_par_lamda[3, :])], tol=1e-10, max_iter=1000, threshold=threshold)
                        weight_lasso_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
                        weight_lasso_coef.append(weight_lasso_model.coef_.tolist()[0])
                        weight_lasso_intercept.append(weight_lasso_model.intercept_)
                        for a in weight_lasso_model.coef_[:, p1 + p2:].tolist()[0]:
                            if abs(a) < 1e-6:
                                weight_lasso_tn += 1
                        for a in weight_lasso_model.coef_[:, p1 + p2:].tolist()[0]:
                            if abs(a) > 1e-6:
                                weight_lasso_fp += 1
                        for a in weight_lasso_model.coef_[:, :p1 + p2].tolist()[0]:
                            if abs(a) < 1e-6:
                                weight_lasso_fn += 1
                        for a in weight_lasso_model.coef_[:, :p1 + p2].tolist()[0]:
                            if abs(a) > 1e-6:
                                weight_lasso_tp += 1
                        y_weight_lasso = weight_lasso_model.predict(XX_test)
                        weight_lasso_loss.append(metrics.mean_squared_error(yy_test, y_weight_lasso) ** 0.5)

                    ########################lsq+weight+lasso####################################
                    gwr_bw = bw[np.argmin(loss_par_bw[1, :])]
                    Aerfa = Kernel(g_coords, gwr_bw, 'bi-square')  # 空间矩阵计算
                    for loc in range(len(g_coords)):
                        aerfa = Aerfa.iloc[:, loc]
                        XX_test, yy_test = add_intercept(X_test_o[list((X_test_o[:, 0] == g_coords[loc][0]) & (X_test_o[:, 1] == g_coords[loc][1])), :][:,2:]), \
                                                         y_test_o[list((X_test_o[:, 0] == g_coords[loc][0]) & (X_test_o[:, 1] == g_coords[loc][1]))]  # 测试的时候使用该地点的数据进行测试
                        if len(XX_test) == 0:  # 如果该地点没有数据就不再进行预测
                            continue
                        #########################进行单地点最小二乘回归###################################
                        lsq_model = one_point_estimation(add_weight=False, lamd=0, tol=1e-10, max_iter=1000, threshold=threshold)
                        lsq_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
                        lsq_coef.append(lsq_model.coef_.tolist()[0])
                        lsq_intercept.append(lsq_model.intercept_)
                        y_lsq = lsq_model.predict(XX_test)
                        lsq_loss.append(metrics.mean_squared_error(yy_test, y_lsq) ** 0.5)
                        ##########################进行局部加权回归##################################
                        weight_model = one_point_estimation(add_weight=True, lamd=0, tol=1e-10, max_iter=1000, threshold=threshold)
                        weight_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
                        weight_coef.append(weight_model.coef_.tolist()[0])
                        weight_intercept.append(weight_model.intercept_)
                        y_weight = weight_model.predict(XX_test)
                        weight_loss.append(metrics.mean_squared_error(yy_test, y_weight) ** 0.5)
                        ########################Lasso##################################
                        lasso_model = one_point_estimation(add_weight=False,  lamd=lamda[np.argmin(loss_par_lamda[2, :])], tol=1e-10, max_iter=1000, threshold=threshold)  # lamd为0时进行局部加权回归
                        lasso_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
                        lasso_coef.append(lasso_model.coef_.tolist()[0])
                        lasso_intercept.append(lasso_model.intercept_)
                        for a in lasso_model.coef_[:, p1 + p2:].tolist()[0]:
                            if abs(a) < 1e-6:
                                lasso_tn += 1
                        for a in lasso_model.coef_[:, p1 + p2:].tolist()[0]:
                            if abs(a) > 1e-6:
                                lasso_fp += 1
                        for a in lasso_model.coef_[:, :p1 + p2].tolist()[0]:
                            if abs(a) < 1e-6:
                                lasso_fn += 1
                        for a in lasso_model.coef_[:, :p1 + p2].tolist()[0]:
                            if abs(a) > 1e-6:
                                lasso_tp += 1
                        y_lsq_lasso = lasso_model.predict(XX_test)
                        lasso_loss.append(metrics.mean_squared_error(yy_test, np.array(y_lsq_lasso)) ** 0.5)

                    print('***********系数估计误差***********')
                    lsq_coef_loss.append(np.sum(np.absolute(np.array(lsq_coef)[:, :p1+p2] - beta[:, :p1+p2])) + np.sum(
                        np.absolute(np.array(lsq_coef)[:, p1+p2:])) + np.sum(
                        np.absolute(np.array(lsq_intercept) - beta[:, p1+p2])))
                    weight_coef_loss.append(np.sum(np.absolute(np.array(weight_coef)[:, :p1+p2] - beta[:, :p1+p2])) + np.sum(
                        np.absolute(np.array(weight_coef)[:, p1+p2:])) + np.sum(
                        np.absolute(np.array(weight_intercept) - beta[:, p1+p2])))
                    lasso_coef_loss.append(np.sum(np.absolute(np.array(lasso_coef)[:, :p1+p2] - beta[:, :p1+p2])) + np.sum(
                        np.absolute(np.array(lasso_coef)[:, p1+p2:])) + np.sum(
                        np.absolute(np.array(lasso_intercept) - beta[:, p1+p2])))
                    weight_lasso_coef_loss.append(
                        np.sum(np.absolute(np.array(weight_lasso_coef)[:, :p1+p2] - beta[:, :p1+p2])) + np.sum(
                            np.absolute(np.array(weight_lasso_coef)[:, p1+p2:])) + np.sum(
                            np.absolute(np.array(weight_lasso_intercept) - beta[:, p1+p2])))
                    sim_coef_loss.append(np.sum(
                        np.absolute(
                            np.array(simultaneous_model.coef_.reshape((l, p)))[:, :p1+p2] - beta[:, :p1+p2])) + np.sum(
                        np.absolute(np.array(simultaneous_model.coef_.reshape((l, p)))[:, p1+p2:])) +
                            np.sum(np.absolute(np.array(simultaneous_model.intercept_) - beta[:, p1+p2].reshape(-1, 1))))
                except:
                    continue

                print("Simulation", repeat + 1, "lsqcl, weightcl, lassocl, weight_lassocl, simcl", end='')
                print(np.average(lsq_coef_loss), np.average(weight_coef_loss), np.average(lasso_coef_loss),
                      np.average(weight_lasso_coef_loss),
                      np.average(sim_coef_loss))
                print("Simulation", repeat + 1, ':lsq_loss', np.average(lsq_loss))
                print("Simulation", repeat + 1, ':weight_loss', np.average(weight_loss))
                print("Simulation", repeat + 1, ':lsq_lasso_loss', np.average(lasso_loss))
                print("Simulation", repeat + 1, ':weight_lasso_loss', np.average(weight_lasso_loss))
                print("Simulation", repeat + 1, ':simultaneous_loss', np.average(sim_loss))
                print("")
                print("lsq_lasso_n1, weight_lasso_n1, sim_n1=", end='')
                print(round(lasso_tn), round(weight_lasso_tn), round(sim_tn))
                print("lsq_lasso_n2, weight_lasso_n2, sim_n2=", end='')
                print(round(lasso_fp), round(weight_lasso_fp), round(sim_fp))
                print("lsq_lasso_n3, weight_lasso_n3, sim_n3=", end='')
                print(round(lasso_fn), round(weight_lasso_fn), round(sim_fn))
                print("lsq_lasso_n4, weight_lasso_n4, sim_n4=", end='')
                print(round(lasso_tp), round(weight_lasso_tp), round(sim_tp))
                loss[0, repeat] = np.average(lsq_loss)
                loss[1, repeat] = np.average(weight_loss)
                loss[2, repeat] = np.average(lasso_loss)
                loss[3, repeat] = np.average(weight_lasso_loss)
                loss[4, repeat] = np.average(sim_loss)
                coef_loss[0, repeat] = np.average(lsq_coef_loss)
                coef_loss[1, repeat] = np.average(weight_coef_loss)
                coef_loss[2, repeat] = np.average(lasso_coef_loss)
                coef_loss[3, repeat] = np.average(weight_lasso_coef_loss)
                coef_loss[4, repeat] = np.average(sim_coef_loss)
                coef_selection_n1[0, repeat] = lasso_tn
                coef_selection_n2[0, repeat] = lasso_fp
                coef_selection_n3[0, repeat] = lasso_fn
                coef_selection_n4[0, repeat] = lasso_tp
                coef_selection_n1[1, repeat] = weight_lasso_tn
                coef_selection_n2[1, repeat] = weight_lasso_fp
                coef_selection_n3[1, repeat] = weight_lasso_fn
                coef_selection_n4[1, repeat] = weight_lasso_tp
                coef_selection_n1[2, repeat] = sim_tn
                coef_selection_n2[2, repeat] = sim_fp
                coef_selection_n3[2, repeat] = sim_fn
                coef_selection_n4[2, repeat] = sim_tp
            average = loss.mean(axis=1)
            std = loss.std(axis=1)
            coef_average = coef_loss.mean(axis=1)
            coef_std = coef_loss.std(axis=1)

            print('lsq_average_loss', round(average[0],3), "(", round(std[0],3), ")")
            print('weight_average_loss', round(average[1],3), "(", round(std[1],3), ")")
            print('lasso_average_loss', round(average[2],3), "(", round(std[2],3), ")")
            print('weight_lasso_average_loss', round(average[3],3), "(", round(std[3],3), ")")
            print('simultaneous_average_loss', round(average[4],3), "(", round(std[4],3), ")")
            print('lsq_coef_average_loss', round(coef_average[0], 3), "(", round(coef_std[0], 3), ")")
            print('weight_coef_average_loss', round(coef_average[1], 3), "(", round(coef_std[1], 3), ")")
            print('lasso_coef_average_loss', round(coef_average[2], 3), "(", round(coef_std[2], 3), ")")
            print('weight_lasso_coef_average_loss', round(coef_average[3], 3), "(", round(coef_std[3], 3), ")")
            print('simultaneous_coef_average_loss', round(coef_average[4], 3), "(", round(coef_std[4], 3), ")")

            print("lsq_lasso_tpr, weight_lasso_tpr, sim_tpr=", end='')
            print(round(sum(coef_selection_n4[0, :]) / ((p1 + p2) * l * repeat_number), 3),
                  round(sum(coef_selection_n4[1, :]) / ((p1 + p2) * l * repeat_number), 3),
                  round(sum(coef_selection_n4[2, :]) / ((p1 + p2) * l * repeat_number), 3))
            print("lsq_lasso_fpr, weight_lasso_fpr, sim_fpr=", end='')
            print(round(sum(coef_selection_n2[0, :]) / (p_noise * l * repeat_number), 3),
                  round(sum(coef_selection_n2[1, :]) / (p_noise * l * repeat_number), 3),
                  round(sum(coef_selection_n2[2, :]) / (p_noise * l * repeat_number), 3))


            # ###################将结果保存到文件中#################################
            output_dir = 'pre/SIM/'
            if output_dir[-1] != "/":
                output_dir += "/"
            if not os.path.isdir(output_dir):
                print("Directory doesn't exist, creating it")
                os.mkdir(output_dir)
            #
            # pd.DataFrame(np.array(weight_lasso_coef)).to_excel(
            #     output_dir + "GWL_" + 'setting=' + setting + ',p1, p2, p_noise=' + str(p1) +','+ str(p2) +','+ str(p_noise) + ',n=' + str(n) + ".xlsx")
            # pd.DataFrame(simultaneous_model.coef_.reshape(l, p)).to_excel(
            #     output_dir + "GWGPL_" + 'setting=' + setting + ',p1, p2, p_noise=' + str(p1) + ',' + str(p2) + ',' + str(
            #         p_noise) + ',n=' + str(n) + ".xlsx")

            result = {}
            result['lsq_average_loss(var)'] = [round(average[0], 3), round(std[0], 3),0]
            result['weight_average_loss(var)'] = [round(average[1], 3), round(std[1], 3),0]
            result['lasso_average_loss(var)'] = [round(average[2], 3), round(std[2], 3),0]
            result['weight_lasso_average_loss(var)'] = [round(average[3], 3), round(std[3], 3),0]
            result['simultaneous_average_loss(var)'] = [round(average[4], 3), round(std[4], 3),0]
            result['lsq_coef_average_loss(var)'] = [round(coef_average[0], 3), round(coef_std[0], 3), 0]
            result['weight_coef_average_loss(var)'] = [round(coef_average[1], 3), round(coef_std[1], 3), 0]
            result['lasso_coef_average_loss(var)'] = [round(coef_average[2], 3), round(coef_std[2], 3), 0]
            result['weight_lasso_coef_average_loss(var)'] = [round(coef_average[3], 3), round(coef_std[3], 3), 0]
            result['simultaneous_coef_average_loss(var)'] = [round(coef_average[4], 3), round(coef_std[4], 3), 0]
            result['lasso_tpr, weight_lasso_tpr, sim_tpr'] = [
                round(sum(coef_selection_n4[0, :]) / ((p1 + p2) * l * repeat_number), 3)
                , round(sum(coef_selection_n4[1, :]) / ((p1 + p2) * l * repeat_number), 3)
                , round(sum(coef_selection_n4[2, :]) / ((p1 + p2) * l * repeat_number), 3)]
            result['lasso_fpr, weight_lasso_fpr, sim_fpr'] = [
                round(sum(coef_selection_n2[0, :]) / (p_noise * l * repeat_number), 3),
                round(sum(coef_selection_n2[1, :]) / (p_noise * l * repeat_number), 3),
                round(sum(coef_selection_n2[2, :]) / (p_noise * l * repeat_number), 3)]
            pd.DataFrame(result).to_excel(
                output_dir + "Result, " + 'setting=' + setting + ',p1, p2, p_noise=' + str(p1) +','+ str(p2) +','+ str(p_noise) +
                'rho_1, rho_2, sigma=' + str(rho_1) +','+ str(rho_2) +','+ str(sigma) + ',n=' + str(sample_average) + ',l=' + str(l) + ".xlsx")