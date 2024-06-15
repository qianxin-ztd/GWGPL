#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2024/1/17 16:31
@Author  : Qian xin
@File    : real.py
"""

from util import *
RANDOM_STATE = 42
import os
from GWGPL import GWGPL
from Compared_method import one_point_estimation
from sklearn import preprocessing, metrics
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")

def parallel_training(index, hyperparam_index, lamda, bw, g_coords):
    """

    :param index:
    :param hyperparam_index: hyperparamter_grid
    :param lamda:
    :param bw:
    :param g_coords:
    :return:
    """
    i, j = hyperparam_index['lamda'], hyperparam_index['bw']
    lamd, gwr_bw = lamda[i], bw[j]
    print("lamd=", lamd, "bw=", gwr_bw, ",calcaulating...")
    Aerfa = Kernel(g_coords, gwr_bw, 'bi-square')  # 空间矩阵计算
    lsq_loss, weight_loss, lsq_lasso_loss, weight_lasso_loss, sim_loss = [], [], [], [], []  # 用来存储每个方法在每个地点的误差
    simultaneous_model = GWGPL(lamd=lamd, tol=1e-10, max_iter=1000)
    simultaneous_model.fit(X_train_x, y_train_x, Aerfa, g_coords, p, l)
    for loc in range(len(g_coords)):
        aerfa = Aerfa.iloc[:, loc]
        XX_test, yy_test = add_intercept(
            X_test_c[list((X_test_c[:, 0] == g_coords[loc][0]) & (X_test_c[:, 1] == g_coords[loc][1])), :][:, 2:]), \
            y_test_c[list(
                (X_test_c[:, 0] == g_coords[loc][0]) & (X_test_c[:, 1] == g_coords[loc][1]))]  # 测试的时候使用该地点的数据进行测试
        if len(XX_test) == 0:  # 如果该地点没有数据就不再进行预测
            continue
        #########################进行单地点最小二乘回归###################################
        if i == 0 and j == 0:  # OLS只执行一次
            lsq_model = one_point_estimation(add_weight=False, lamd=0, tol=1e-10, max_iter=1000)
            lsq_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
            y_lsq = lsq_model.predict(XX_test)
            lsq_loss.append(metrics.mean_squared_error(yy_test, y_lsq) ** 0.5)
        ##########################进行局部加权回归##################################
        weight_model = one_point_estimation(add_weight=True, lamd=0, tol=1e-10, max_iter=1000)
        weight_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
        y_weight = weight_model.predict(XX_test)
        weight_loss.append(metrics.mean_squared_error(yy_test, y_weight) ** 0.5)
        # ########################进行最小二乘+Lasso##################################
        lasso_model = one_point_estimation(add_weight=False, lamd=lamd, tol=1e-10, max_iter=1000)
        lasso_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
        y_lsq_lasso = lasso_model.predict(XX_test)
        lsq_lasso_loss.append(metrics.mean_squared_error(yy_test, np.array(y_lsq_lasso)) ** 0.5)
        #######################进行局部加权+Lasso##################################
        weight_lasso_model = one_point_estimation(add_weight=True, lamd=lamd, tol=1e-10, max_iter=1000)
        weight_lasso_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
        y_weight_lasso = weight_lasso_model.predict(XX_test)
        weight_lasso_loss.append(metrics.mean_squared_error(yy_test, y_weight_lasso) ** 0.5)
        ##############################进行整体同时估计################################
        y_sim = simultaneous_model.predict(XX_test, loc, p)
        sim_loss.append(metrics.mean_squared_error(yy_test, np.array(y_sim)) ** 0.5)
    return [i, j,  np.average(lsq_loss) if len(lsq_loss) != 0 else 0,
                   np.average(weight_loss) if len(weight_loss) != 0 else 0,
                   np.average(lsq_lasso_loss) if len(lsq_lasso_loss) != 0 else 0,
                   np.average(weight_lasso_loss) if len(weight_lasso_loss) != 0 else 0,
                   np.average(sim_loss) if len(sim_loss) != 0 else 0]

def Subsite_standardization(data, g_coords):  # 分地区进行标准化
    zscore = preprocessing.StandardScaler()
    for i in range(len(g_coords)):
        index = data.index[(data.iloc[:, 0] == g_coords[i, 0])].tolist()
        data.iloc[index, 2:] = pd.DataFrame(zscore.fit_transform(data.iloc[index, 2:]))  # z-score标准化
    return data


if __name__ == '__main__':
    # # 读取csv数据
    data = pd.read_excel('./data/data_select1.xlsx').iloc[:, 2:]
    # coordinate = pd.read_excel('./data/coordinate.xlsx')
    # map_make(data, coordinate)  # 得到样本在地图上的分布
    data[['longitude', 'latitude']] = coord_limit(np.array(data[['longitude', 'latitude']]))
    g_coords = np.unique(np.array(data[['longitude', 'latitude']]), axis=0)
    # plt.scatter(g_coords[:, 0], g_coords[:, 1])
    # plt.show()
    zscore = preprocessing.StandardScaler()
    data.iloc[:, 2:] = zscore.fit_transform(data.iloc[:, 2:])  # z-score标准化
    data = Subsite_standardization(data, g_coords)

    p = data.shape[1] - 3  # the number of covariates
    l = len(g_coords)  # the number of location

    b_min, b_max = bw_selection(g_coords)
    bw = np.linspace(b_max / 6, b_max, 10)  # 提前给定的bw的取值
    lamda = np.linspace(1, 30, 30)
    loss_par = np.zeros([5, len(lamda) * len(bw)])
    split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_STATE)
    split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_STATE)
    for train_index, test_index in split1.split(data, data["longitude"]):
        train_o, test_o = data.iloc[train_index, :], data.iloc[test_index, :]
        X_train_o, y_train_o = train_o.iloc[:, 0:p + 2], train_o.iloc[:, p + 2]
        X_test_o, y_test_o = test_o.iloc[:, 0:p + 2], test_o.iloc[:, p + 2]
    for train_index, test_index in split2.split(train_o, train_o["longitude"]):
        train_c, test_c = train_o.iloc[train_index, :], train_o.iloc[test_index, :]
        X_train_x, y_train_x = train_c.iloc[:, 0:p + 2], train_c.iloc[:, p + 2]
        X_test_c, y_test_c = test_c.iloc[:, 0:p + 2], test_c.iloc[:, p + 2]
    X_train_x, X_test_c, X_test_o, y_train_x, y_test_c, y_test_o = np.array(X_train_x), np.array(X_test_c), \
        np.array(X_test_o), np.array(y_train_x), np.array(y_test_c), np.array(y_test_o)

    model_ls = []
    param_range = {'lamda': range(len(lamda)), 'bw': range(len(bw))}
    param_grid = ParameterGrid(param_range)  # 超参数索引网格
    with Parallel(n_jobs=1) as parallel:
       model_ls = parallel(delayed(parallel_training)
                       (index, hyperparam_index, lamda, bw, g_coords
                        ) for index, hyperparam_index in enumerate(param_grid))

    for i ,j , lsq_loss, weight_loss, lasso_loss, weight_lasso_loss, sim_loss in model_ls:
        loss_par[0, i * len(bw) + j],loss_par[1, i * len(bw) + j],loss_par[2, i * len(bw) + j],\
            loss_par[3, i * len(bw) + j],loss_par[4, i * len(bw) + j] = lsq_loss, weight_loss, lasso_loss, weight_lasso_loss, sim_loss

    print('###############交叉验证结束，输出此时每个模型的最优参数######################')
    print("Weight_model (The best bw):", bw[np.argmin(loss_par[1, :]) % len(bw)])
    print("Lasso_model (The best lamda):", lamda[np.argmin(loss_par[2, :]) // len(bw)])
    print("Weight_Lasso_model (The best lamda):", lamda[np.argmin(loss_par[3, :]) // len(bw)],
          "(The best bw):", bw[np.argmin(loss_par[3, :]) % len(bw)])
    print("simultaneous_model (The best lamda):", lamda[np.argmin(loss_par[4, :]) // len(bw)],
          "(The best bw):", bw[np.argmin(loss_par[4, :]) % len(bw)])

    print('###############开始进行测试集测试########################')
    lsq_loss, weight_loss, lsq_lasso_loss, weight_lasso_loss, sim_loss = [], [], [], [], []  # 用来存储每个方法在每个地点的误差
    lsq_coef, weight_coef, lasso_coef, weight_lasso_coef = [], [], [], []
    lsq_intercept, weight_intercept, lasso_intercept, weight_lasso_intercept = [], [], [], []
    lasso_sp, weight_lasso_sp, sim_sp = 0, 0, 0
    ########################Group+GWR#############################
    gwr_bw = bw[np.argmin(loss_par[4, :]) % len(bw)]
    Aerfa = Kernel(g_coords, gwr_bw, 'bi-square')  # 空间矩阵计算
    simultaneous_model = GWGPL(lamd=lamda[np.argmin(loss_par[4, :]) // len(bw)], tol=1e-10, max_iter=1000)
    simultaneous_model.fit(X_train_x, y_train_x, Aerfa, g_coords, p, l)
    for lst in simultaneous_model.coef_.reshape((p, l)).T.tolist():
        for a in lst:
            if abs(a) < 1e-6:
                sim_sp += 1
    for loc in range(len(g_coords)):
        XX_test, yy_test = add_intercept(
            X_test_o[list((X_test_o[:, 0] == g_coords[loc][0]) & (X_test_o[:, 1] == g_coords[loc][1])), :][:, 2:]), \
            y_test_o[list((X_test_o[:, 0] == g_coords[loc][0]) & (X_test_o[:, 1] == g_coords[loc][1]))]  # 测试的时候使用该地点的数据进行测试
        if len(XX_test) == 0:  # 如果该地点没有数据就不再进行预测
            continue
        # ##############################进行整体同时估计################################
        y_sim = simultaneous_model.predict(XX_test, loc, p)
        sim_loss.append(metrics.mean_squared_error(yy_test, np.array(y_sim)) ** 0.5)


    ########################lasso+GWR###############################
    gwr_bw = bw[np.argmin(loss_par[3, :]) % len(bw)]
    Aerfa = Kernel(g_coords, gwr_bw, 'bi-square')  # 空间矩阵计算
    for loc in range(len(g_coords)):
        aerfa = Aerfa.iloc[:, loc]
        XX_test, yy_test = add_intercept(
            X_test_o[list((X_test_o[:, 0] == g_coords[loc][0]) & (X_test_o[:, 1] == g_coords[loc][1])), :][:, 2:]), \
            y_test_o[list((X_test_o[:, 0] == g_coords[loc][0]) & (X_test_o[:, 1] == g_coords[loc][1]))]  # 测试的时候使用该地点的数据进行测试
        if len(XX_test) == 0:  # 如果该地点没有数据就不再进行预测
            continue
        # #######################进行局部加权+Lasso##################################
        weight_lasso_model = one_point_estimation(add_weight=True,lamd=lamda[np.argmin(loss_par[3, :]) // len(bw)],
                                                  tol=1e-10, max_iter=1000)
        weight_lasso_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
        weight_lasso_coef.append(weight_lasso_model.coef_.tolist()[0])
        weight_lasso_intercept.append(weight_lasso_model.intercept_)
        for a in weight_lasso_model.coef_.tolist()[0]:
            if abs(a) < 1e-6:
                weight_lasso_sp += 1
        y_weight_lasso = weight_lasso_model.predict(XX_test)
        weight_lasso_loss.append(metrics.mean_squared_error(yy_test, y_weight_lasso) ** 0.5)

    ########################lsq+weight+lasso####################################
    gwr_bw = bw[np.argmin(loss_par[1, :]) % len(bw)]
    Aerfa = Kernel(g_coords, gwr_bw, 'bi-square')  # 空间矩阵计算
    for loc in range(len(g_coords)):
        aerfa = Aerfa.iloc[:, loc]
        XX_test, yy_test = add_intercept(
            X_test_o[list((X_test_o[:, 0] == g_coords[loc][0]) & (X_test_o[:, 1] == g_coords[loc][1])), :][:, 2:]), \
            y_test_o[list((X_test_o[:, 0] == g_coords[loc][0]) & (X_test_o[:, 1] == g_coords[loc][1]))]  # 测试的时候使用该地点的数据进行测试
        if len(XX_test) == 0:  # 如果该地点没有数据就不再进行预测
            continue
        #########################进行单地点最小二乘回归###################################
        lsq_model = one_point_estimation(add_weight=False, lamd=0, tol=1e-10, max_iter=1000)
        lsq_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
        lsq_coef.append(lsq_model.coef_.tolist()[0])
        lsq_intercept.append(lsq_model.intercept_[0])
        y_lsq = lsq_model.predict(XX_test)
        lsq_loss.append(metrics.mean_squared_error(yy_test, y_lsq) ** 0.5)
        ##########################进行局部加权回归##################################
        weight_model = one_point_estimation(add_weight=True, lamd=0, tol=1e-10, max_iter=1000)  # lamd为0时进行局部加权回归
        weight_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
        weight_coef.append(weight_model.coef_.tolist()[0])
        weight_intercept.append(weight_model.intercept_[0])
        y_weight = weight_model.predict(XX_test)
        weight_loss.append(metrics.mean_squared_error(yy_test, y_weight) ** 0.5)
        ########################Lasso##################################
        lasso_model = one_point_estimation(add_weight=False, lamd=lamda[np.argmin(loss_par[2, :]) // len(bw)], tol=1e-10,
                                           max_iter=1000)
        lasso_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
        lasso_coef.append(lasso_model.coef_.tolist()[0])
        lasso_intercept.append(lasso_model.intercept_)
        for a in lasso_model.coef_.tolist()[0]:
            if abs(a) < 1e-6:
                lasso_sp += 1
        y_lsq_lasso = lasso_model.predict(XX_test)
        lsq_lasso_loss.append(metrics.mean_squared_error(yy_test, np.array(y_lsq_lasso)) ** 0.5)

    print('lsq_average_loss', np.average(lsq_loss))
    print('weight_average_loss', np.average(weight_loss))
    print('lsq_lasso_average_loss', np.average(lsq_lasso_loss))
    print('weight_lasso_average_loss', np.average(weight_lasso_loss))
    print('simultaneous_average_loss', np.average(sim_loss))

    print("lsq_lasso_tn, weight_lasso_tn, sim_tn=", end='')
    print(round(lasso_sp / (l * p), 3), round(weight_lasso_sp / (l * p), 3), round(sim_sp / (l * p), 3))

    print('系数保存中.....')
    lsq_coef, weight_coef, lasso_coef, weight_lasso_coef = np.array(lsq_coef), np.array(weight_coef), np.array(
        lasso_coef), np.array(weight_lasso_coef)
    lsq_intercept, weight_intercept, lasso_intercept, weight_lasso_intercept = np.array(lsq_intercept), np.array(
        weight_intercept), np.array(
        lasso_intercept), np.array(weight_lasso_intercept)
    sim_coef, sim_intercept = np.array(simultaneous_model.coef_.reshape((l, p))), np.array(
        simultaneous_model.intercept_)
    lsq_fin = np.hstack((lsq_coef, lsq_intercept.reshape(-1, 1)))
    weight_fin = np.hstack((weight_coef, weight_intercept.reshape(-1, 1)))
    lasso_fin = np.hstack((lasso_coef, lasso_intercept.reshape(-1, 1)))
    weight_lasso_fin = np.hstack((weight_lasso_coef, weight_lasso_intercept.reshape(-1, 1)))
    sim_fin = np.hstack((sim_coef, sim_intercept.reshape(-1, 1)))

    output_dir = 'estimated_coef/'
    if output_dir[-1] != "/":
        output_dir += "/"
    if not os.path.isdir(output_dir):
        print("Directory doesn't exist, creating it")
        os.mkdir(output_dir)
    pd.DataFrame(lsq_fin).to_excel(output_dir + "Coef_lsq" + ".xlsx")
    pd.DataFrame(weight_fin).to_excel(output_dir + "Coef_weight" + ".xlsx")
    pd.DataFrame(lasso_fin).to_excel(output_dir + "Coef_lasso" + ".xlsx")
    pd.DataFrame(weight_lasso_fin).to_excel(output_dir + "Coef_weight_lasso" + ".xlsx")
    pd.DataFrame(sim_fin).to_excel(output_dir + "Coef_sim" + ".xlsx")