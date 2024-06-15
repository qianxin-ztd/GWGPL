import numpy as np
import pandas as pd
import math
from math import sqrt
import pyproj
from sklearn.linear_model import LinearRegression

def bw_selection(g_coords):
    """
    get bw_min and bw_max
    :param g_coords:
    :return:
    """
    bb = np.zeros((len(g_coords), len(g_coords)))
    for i in range(len(g_coords)):
        for j in range(len(g_coords)):
            bb[i][j] = pow(pow(g_coords[i][0] - g_coords[j][0], 2)+pow(g_coords[i][1] - g_coords[j][1], 2), 0.5)
    return np.min(bb), np.max(bb)


def add_intercept(X):
    """
    add intercept to X
    :param X:
    :return: [X, X_0]
    """
    return np.c_[X, np.ones(len(X))]

def Kernel(g_coords, gwr_bw, kernel):  # 定义核函数
    """
    Kernel_function
    :param g_coords: [longitude, latitude]
    :param gwr_bw:  bandwidth
    :param kernel:  kernel function
    :return:  weight matrix
    """
    aerfa = np.zeros((len(g_coords), len(g_coords)))
    for i in range(len(g_coords)):
        for j in range(len(g_coords)):
            aerfa[i][j] = pow(pow(g_coords[i][0] - g_coords[j][0], 2)+pow(g_coords[i][1] - g_coords[j][1], 2), 0.5)
            if aerfa[i][j] > gwr_bw:
                aerfa[i][j] = 0
            else:
                if kernel == 'threshold': aerfa[i][j] = 1
                if kernel == 'bi-square': aerfa[i][j] = pow(1 - pow((aerfa[i][j] / gwr_bw), 2), 2)
                if kernel == 'gaussian': aerfa[i][j] = np.exp(-0.5 * pow((aerfa[i][j] / gwr_bw), 2))
                if kernel == 'exponential': aerfa[i][j] = np.exp(-aerfa[i][j] / gwr_bw)
    return pd.DataFrame(aerfa, index=[list(g_coords[:, 0]), list(g_coords[:, 1])],
                         columns=[list(g_coords[:, 0]), list(g_coords[:, 1])])

def WGS84ToGK_Single(coords):  # Converts geographic coordinates to plane coordinates
    """
    WGS84坐标转高斯坐标
    :param lon:  WGS84坐标经度
    :param lat:  WGS84坐标纬度
    :return: 高斯坐标x,y
    """
    lon = coords[:, 0]
    lat = coords[:, 1]
    p1 = pyproj.Proj(init="epsg:4326")
    p2 = pyproj.Proj(init="epsg:3857")
    x1, y1 = p1(lon, lat)
    x2, y2 = pyproj.transform(p1, p2, x1, y1)
    return np.array([x2,y2]).T


def millerToXY(coords):
    """
    经纬度转换为平面坐标系中的x,y 利用米勒坐标系
    :param lon: 经度
    :param lat: 维度
    :return:  高斯坐标x,y
    """
    xy_coordinate = []  # 转换后的XY坐标集
    lon = coords[:, 0]
    lat = coords[:, 1]
    L = 6381372 * math.pi * 2
    W = L
    H = L / 2
    mill = 2.3
    x = lon * math.pi / 180
    y = lat * math.pi / 180
    ylist = []
    for i in range(len(x)):
        ylist.append(1.25 * math.log(math.tan(0.25 * math.pi + 0.4 * y[i])))
    y = np.array(ylist)
    x = (W / 2) + (W / (2 * math.pi)) * x
    y = (H / 2) - (H / (2 * mill)) * y
    return np.array([x, y]).T

def coord_limit(coords):
    """
    将转换为的平面坐标约束到01之间
    :param coords:
    :return:
    """
    coords = WGS84ToGK_Single(coords)
    coords[:, 0] = (coords[:, 0] - np.min(coords[:, 0])) / (np.max(coords[:, 0]) - np.min(coords[:, 0]))
    coords[:, 1] = (coords[:, 1] - np.min(coords[:, 1])) / (np.max(coords[:, 1]) - np.min(coords[:, 1]))
    return coords

def localWeightRegression(X, y, wt):
    """
    Geographic weighted regression
    :param X:
    :param y:
    :param wt: weight
    :return:
    """
    w = np.diag(np.array(wt))
    beta = np.linalg.pinv(X[:, 2:].T @ w @ X[:, 2:]) @ (X[:, 2:].T @ w @ y)
    beta = beta.reshape(len(beta), 1)
    return beta

def Least_squares(X, y, g_coords, loc):  # 最小二乘
    """
    OLS
    :param X:
    :param y:
    :param g_coords:
    :param loc:
    :return:
    """
    X_train_lsq = X[list((X[:, 0] == g_coords[loc][0]) & (X[:, 1] == g_coords[loc][1])), :][:, 2:]
    y_train_lsq = y[list((X[:, 0] == g_coords[loc][0]) & (X[:, 1] == g_coords[loc][1]))]
    return np.linalg.pinv((X_train_lsq.T.dot(X_train_lsq))).dot(X_train_lsq.T).dot(y_train_lsq).reshape(-1, 1)