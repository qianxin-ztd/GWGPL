from Compared_method import one_point_estimation
from sklearn.model_selection import train_test_split
from sklearn import metrics
import random
from util import *
RANDOM_STATE = 42
from GWGPL import GWGPL
import os

def generate_one_point_data(N, beta, setting):   # 生成一个地点的数据
    # setting为0表示全部为离散变量，setting为1表示一半离散变量，一半连续变量，setting为2表示全部为连续变量
    if setting == 1:
        # z1 = np.random.multivariate_normal(np.zeros(m1), np.identity(m1), N)
        z2 = np.random.multivariate_normal(np.zeros(p2), np.identity(p2), N)
        z3 = np.random.multivariate_normal(np.zeros(p_noise), np.identity(p_noise), N)
        # z1[z1 >= 1], z1[z1 <= -1], z2[z2 >= 1], z2[z2 <= -1], z3[z3 >= 1], z3[z3 <= -1] = 1, -1, 1, -1, 1, -1
        # z1 = np.random.uniform(-1, 1, m1 * N).reshape(N, m1)
        # z2 = np.random.uniform(-1, 1, m2 * N).reshape(N, m2)
        # z3 = np.random.uniform(-1, 1, m_noise * N).reshape(N, m_noise)
        # 通过分箱方法将连续变量进行离散化
        for i in range(5):
            z2[:, i] = np.digitize(z2[:, i], [np.quantile(z2[:, i], 0.25)])  #比0.25分点小的返回值为0，其余值返回为1
        # for i in range(5, 10):
        #     z2[:, i] = np.digitize(z2[:, i], [np.quantile(z2[:, i], 0.5)])
        # for i in range(10, 15):
        #     z2[:, i] = np.digitize(z2[:, i], [np.quantile(z2[:, i], 0.75)])
        # data = np.concatenate([z1, z2], axis=1)   # 变量数据集
        data = z2
        # 加上误差项，每一个误差项服从多元正态分布，且每一个一元正态分布相互独立
        y = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :] / 3
        # y_1 = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :]
        # y_2 = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :] / 2
        # y_3 = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :] / 3
        # y_4 = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :] / 4
        # plt.plot(list(range(len(y))), y, label='ori')
        # plt.plot(list(range(len(y_1))), y_1, label='y_1')
        # plt.plot(list(range(len(y_2))), y_2, label='y_2')
        # plt.plot(list(range(len(y_3))), y_3, label='y_3')
        # plt.plot(list(range(len(y_4))), y_4, label='y_4')
        # plt.show()
        data = np.concatenate([data, z3], axis=1)  # 原始数据集+噪声数据集
        dataset = np.concatenate((data.astype(np.float32), y.astype(np.float32).reshape(-1, 1)), axis=1)
        dataset = pd.DataFrame(dataset, columns=[*[f'x{i+1}' for i in range(data.shape[1])], 'y'])
    elif setting == 2:
        z1 = np.random.multivariate_normal(np.zeros(p1), np.identity(p1), N)
        z2 = np.random.multivariate_normal(np.zeros(p2), np.identity(p2), N)
        z3 = np.random.multivariate_normal(np.zeros(p_noise), np.identity(p_noise), N)
        # z1[z1 >= 1], z1[z1 <= -1], z2[z2 >= 1], z2[z2 <= -1], z3[z3 >= 1], z3[z3 <= -1] = 1, -1, 1, -1, 1, -1
        # z1 = np.random.uniform(-1, 1, m1 * N).reshape(N, m1)
        # z2 = np.random.uniform(-1, 1, m2 * N).reshape(N, m2)
        # z3 = np.random.uniform(-1, 1, m_noise * N).reshape(N, m_noise)
        # 通过分箱方法将连续变量进行离散化
        for i in range(2):
            z2[:, i] = np.digitize(z2[:, i], [np.quantile(z2[:, i], 0.5)])  # 比0.25分点小的返回值为0，其余值返回为1
        # for i in range(5, 10):
        #     z2[:, i] = np.digitize(z2[:, i], [np.quantile(z2[:, i], 0.5)])
        # for i in range(10, 15):
        #     z2[:, i] = np.digitize(z2[:, i], [np.quantile(z2[:, i], 0.75)])
        data = np.concatenate([z1, z2], axis=1)   # 变量数据集
        # data = z2
        # 加上误差项，每一个误差项服从多元正态分布，且每一个一元正态分布相互独立
        y = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0,
                                                     :] / 3
        # y_1 = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :]
        # y_2 = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :] / 2
        # y_3 = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :] / 3
        # y_4 = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :] / 4
        # plt.plot(list(range(len(y))), y, label='ori')
        # plt.plot(list(range(len(y_1))), y_1, label='y_1')
        # plt.plot(list(range(len(y_2))), y_2, label='y_2')
        # plt.plot(list(range(len(y_3))), y_3, label='y_3')
        # plt.plot(list(range(len(y_4))), y_4, label='y_4')
        # plt.show()
        data = np.concatenate([data, z3], axis=1)  # 原始数据集+噪声数据集
        dataset = np.concatenate((data.astype(np.float32), y.astype(np.float32).reshape(-1, 1)), axis=1)
        dataset = pd.DataFrame(dataset, columns=[*[f'x{i + 1}' for i in range(data.shape[1])], 'y'])
    elif setting == 3:
        z1 = np.random.multivariate_normal(np.zeros(p1), np.identity(p1), N)
        # z2 = np.random.multivariate_normal(np.zeros(m2), np.identity(m2), N)
        z3 = np.random.multivariate_normal(np.zeros(p_noise), np.identity(p_noise), N)
        # z1[z1 >= 1], z1[z1 <= -1], z2[z2 >= 1], z2[z2 <= -1], z3[z3 >= 1], z3[z3 <= -1] = 1, -1, 1, -1, 1, -1
        # z1 = np.random.uniform(-1, 1, m1 * N).reshape(N, m1)
        # z2 = np.random.uniform(-1, 1, m2 * N).reshape(N, m2)
        # z3 = np.random.uniform(-1, 1, m_noise * N).reshape(N, m_noise)
        # 通过分箱方法将连续变量进行离散化
        # for i in range(5):
        #     z2[:, i] = np.digitize(z2[:, i], [np.quantile(z2[:, i], 0.25)])  # 比0.25分点小的返回值为0，其余值返回为1
        # for i in range(5, 10):
        #     z2[:, i] = np.digitize(z2[:, i], [np.quantile(z2[:, i], 0.5)])
        # for i in range(10, 15):
        #     z2[:, i] = np.digitize(z2[:, i], [np.quantile(z2[:, i], 0.75)])
        # data = np.concatenate([z1, z2], axis=1)   # 变量数据集
        data = z1
        # 加上误差项，每一个误差项服从多元正态分布，且每一个一元正态分布相互独立
        y = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0,
                                                     :] / 2
        # y_1 = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :]
        # y_2 = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :] / 2
        # y_3 = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :] / 3
        # y_4 = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :] / 4
        # plt.plot(list(range(len(y))), y, label='ori')
        # plt.plot(list(range(len(y_1))), y_1, label='y_1')
        # plt.plot(list(range(len(y_2))), y_2, label='y_2')
        # plt.plot(list(range(len(y_3))), y_3, label='y_3')
        # plt.plot(list(range(len(y_4))), y_4, label='y_4')
        # plt.show()
        data = np.concatenate([data, z3], axis=1)  # 原始数据集+噪声数据集
        dataset = np.concatenate((data.astype(np.float32), y.astype(np.float32).reshape(-1, 1)), axis=1)
        dataset = pd.DataFrame(dataset, columns=[*[f'x{i + 1}' for i in range(data.shape[1])], 'y'])
    elif setting == 4:
        # z1 = np.random.multivariate_normal(np.zeros(m1), np.identity(m1), N)
        z2 = np.random.multivariate_normal(np.zeros(p2), np.identity(p2), N)
        z3 = np.random.multivariate_normal(np.zeros(p_noise), np.identity(p_noise), N)
        # z1[z1 >= 1], z1[z1 <= -1], z2[z2 >= 1], z2[z2 <= -1], z3[z3 >= 1], z3[z3 <= -1] = 1, -1, 1, -1, 1, -1
        # z1 = np.random.uniform(-1, 1, m1 * N).reshape(N, m1)
        # z2 = np.random.uniform(-1, 1, m2 * N).reshape(N, m2)
        # z3 = np.random.uniform(-1, 1, m_noise * N).reshape(N, m_noise)
        # 通过分箱方法将连续变量进行离散化
        for i in range(5):
            z2[:, i] = np.digitize(z2[:, i], [np.quantile(z2[:, i], 0.25)])  #比0.25分点小的返回值为0，其余值返回为1
        for i in range(5, 10):
            z2[:, i] = np.digitize(z2[:, i], [np.quantile(z2[:, i], 0.5)])
        # for i in range(10, 15):
        #     z2[:, i] = np.digitize(z2[:, i], [np.quantile(z2[:, i], 0.75)])
        # data = np.concatenate([z1, z2], axis=1)   # 变量数据集
        data = z2
        # 加上误差项，每一个误差项服从多元正态分布，且每一个一元正态分布相互独立
        y = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :] / 3
        # y_1 = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :]
        # y_2 = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :] / 2
        # y_3 = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :] / 3
        # y_4 = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :] / 4
        # plt.plot(list(range(len(y))), y, label='ori')
        # plt.plot(list(range(len(y_1))), y_1, label='y_1')
        # plt.plot(list(range(len(y_2))), y_2, label='y_2')
        # plt.plot(list(range(len(y_3))), y_3, label='y_3')
        # plt.plot(list(range(len(y_4))), y_4, label='y_4')
        # plt.show()
        data = np.concatenate([data, z3], axis=1)  # 原始数据集+噪声数据集
        dataset = np.concatenate((data.astype(np.float32), y.astype(np.float32).reshape(-1, 1)), axis=1)
        dataset = pd.DataFrame(dataset, columns=[*[f'x{i+1}' for i in range(data.shape[1])], 'y'])
    elif setting == 5:
        z1 = np.random.multivariate_normal(np.zeros(p1), np.identity(p1), N)
        z2 = np.random.multivariate_normal(np.zeros(p2), np.identity(p2), N)
        z3 = np.random.multivariate_normal(np.zeros(p_noise), np.identity(p_noise), N)
        # z1[z1 >= 1], z1[z1 <= -1], z2[z2 >= 1], z2[z2 <= -1], z3[z3 >= 1], z3[z3 <= -1] = 1, -1, 1, -1, 1, -1
        # z1 = np.random.uniform(-1, 1, m1 * N).reshape(N, m1)
        # z2 = np.random.uniform(-1, 1, m2 * N).reshape(N, m2)
        # z3 = np.random.uniform(-1, 1, m_noise * N).reshape(N, m_noise)
        # 通过分箱方法将连续变量进行离散化
        for i in range(5):
            z2[:, i] = np.digitize(z2[:, i], [np.quantile(z2[:, i], 0.25)])  # 比0.25分点小的返回值为0，其余值返回为1
        # for i in range(4, 8):
        #     z2[:, i] = np.digitize(z2[:, i], [np.quantile(z2[:, i], 0.5)])
        # for i in range(10, 15):
        #     z2[:, i] = np.digitize(z2[:, i], [np.quantile(z2[:, i], 0.75)])
        data = np.concatenate([z1, z2], axis=1)   # 变量数据集
        # data = z2
        # 加上误差项，每一个误差项服从多元正态分布，且每一个一元正态分布相互独立
        y = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0,
                                                     :] / 3
        # y_1 = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :]
        # y_2 = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :] / 2
        # y_3 = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :] / 3
        # y_4 = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :] / 4
        # plt.plot(list(range(len(y))), y, label='ori')
        # plt.plot(list(range(len(y_1))), y_1, label='y_1')
        # plt.plot(list(range(len(y_2))), y_2, label='y_2')
        # plt.plot(list(range(len(y_3))), y_3, label='y_3')
        # plt.plot(list(range(len(y_4))), y_4, label='y_4')
        # plt.show()
        data = np.concatenate([data, z3], axis=1)  # 原始数据集+噪声数据集
        dataset = np.concatenate((data.astype(np.float32), y.astype(np.float32).reshape(-1, 1)), axis=1)
        dataset = pd.DataFrame(dataset, columns=[*[f'x{i + 1}' for i in range(data.shape[1])], 'y'])
    elif setting == 6:
        z1 = np.random.multivariate_normal(np.zeros(p1), np.identity(p1), N)
        # z2 = np.random.multivariate_normal(np.zeros(m2), np.identity(m2), N)
        z3 = np.random.multivariate_normal(np.zeros(p_noise), np.identity(p_noise), N)
        # z1[z1 >= 1], z1[z1 <= -1], z2[z2 >= 1], z2[z2 <= -1], z3[z3 >= 1], z3[z3 <= -1] = 1, -1, 1, -1, 1, -1
        # z1 = np.random.uniform(-1, 1, m1 * N).reshape(N, m1)
        # z2 = np.random.uniform(-1, 1, m2 * N).reshape(N, m2)
        # z3 = np.random.uniform(-1, 1, m_noise * N).reshape(N, m_noise)
        # 通过分箱方法将连续变量进行离散化
        # for i in range(5):
        #     z2[:, i] = np.digitize(z2[:, i], [np.quantile(z2[:, i], 0.25)])  # 比0.25分点小的返回值为0，其余值返回为1
        # for i in range(5, 10):
        #     z2[:, i] = np.digitize(z2[:, i], [np.quantile(z2[:, i], 0.5)])
        # for i in range(10, 15):
        #     z2[:, i] = np.digitize(z2[:, i], [np.quantile(z2[:, i], 0.75)])
        # data = np.concatenate([z1, z2], axis=1)   # 变量数据集
        data = z1
        # 加上误差项，每一个误差项服从多元正态分布，且每一个一元正态分布相互独立
        y = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0,
                                                     :] / 3
        # y_1 = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :]
        # y_2 = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :] / 2
        # y_3 = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :] / 3
        # y_4 = np.c_[data, np.ones(len(data))] @ beta + np.random.multivariate_normal(np.zeros(N), np.identity(N), N)[0, :] / 4
        # plt.plot(list(range(len(y))), y, label='ori')
        # plt.plot(list(range(len(y_1))), y_1, label='y_1')
        # plt.plot(list(range(len(y_2))), y_2, label='y_2')
        # plt.plot(list(range(len(y_3))), y_3, label='y_3')
        # plt.plot(list(range(len(y_4))), y_4, label='y_4')
        # plt.show()
        data = np.concatenate([data, z3], axis=1)  # 原始数据集+噪声数据集
        dataset = np.concatenate((data.astype(np.float32), y.astype(np.float32).reshape(-1, 1)), axis=1)
        dataset = pd.DataFrame(dataset, columns=[*[f'x{i + 1}' for i in range(data.shape[1])], 'y'])
    return dataset

def generate_repeat_data(repeat_number, setting):
    beta = np.full((l, p1 + p2 + 1), np.nan)  # 初始化回归系数
    a = np.linspace(1.5, 4, p1 + p2 + 1)  # 得到不同的特征
    for feature in range(p1 + p2 + 1):  # 得到所有地点的回归系数
        beta[:, feature] = (np.sqrt(0.5 - (np.sqrt((g_coords[:, 0] - 0.5) ** 2 + (g_coords[:, 1] - 0.5) ** 2)) ** 2)) / a[feature]  # 得到beta值
    repeat_data = []
    repeat_number = repeat_number
    for aa in range(repeat_number):
        np.random.seed(1314 + aa)
        simdat = []
        for i in range(l):  # 对每一个地点
            datset = generate_one_point_data(n[i], beta[i, :], setting)
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
    return repeat_data, beta


if __name__ == '__main__':
    ############ data generating ##############
    setting = 5
    print(
        'setting = 5, m1, m2, m_noise = 5, 5, 10, n = [147, 178, 164, 38, 92, 102, 200, 193, 200, 82, 62, 119, 100, 142, 37, 165, 63, 45, 194, 81]')
    p1, p2, p_noise = 5, 5, 10  # number of continuous covariates, discrete covariates and noise covariates
    p, l = p1 + p2 + p_noise, 20  # number of covariates and number of locations
    # n = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
    # n = [60, 60, 60, 60, 60, 60, 60, 60, 60, 60]
    # n = [180, 180, 180, 180, 180, 180, 180, 180, 180, 180]
    # n = [300, 300, 300, 300, 300, 300, 300, 300, 300, 300]
    n = [147, 178, 164, 38, 92, 102, 200, 193, 200, 82, 62, 119, 100, 142, 37, 165, 63, 45, 194, 81]
    # n = [143, 88, 36, 42, 62, 77, 55, 56, 135, 57, 191, 112, 43, 39, 51, 101, 81, 143, 179, 103]
    # g_coords = np.array([
    #              [0.02375981, 0.08825333],
    #              [0.03665332, 0.60964299],
    #              [0.32523272, 0.88500807],
    #              [0.08494575, 0.25065348],
    #              [0.74082562, 0.93311949],
    #              [0.90742148, 0.66599465],
    #              [0.93718995, 0.83749397],
    #              [0.92127660, 0.34535785],
    #              [0.73589767, 0.06708488],
    #              [0.52188345, 0.04101815],
    #              [0.29536456, 0.08254615],
    #              [0.56423689, 0.18756423],
    #              [0.89920136, 0.09598782],
    #              [0.78032645, 0.60654632],
    #              [0.82135798, 0.83365454],
    #              [0.60145892, 0.83866978],
    #              [0.45321645, 0.75362456],
    #              [0.16054231, 0.86345789],
    #              [0.12537968, 0.45231654],
    #              [0.27154697, 0.72564324]])

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
        [0.58456213, 0.421536970],
        [0.40562389, 0.479236465],
        [0.42256963, 0.539462346],
        [0.47765852, 0.462351321],
        [0.50946365, 0.569423132],
        [0.52247896, 0.462513645],
        [0.55264352, 0.413254612],
        [0.57512346, 0.567432152],
        [0.60321987, 0.459621321],
        [0.57896546, 0.512646879],
        [0.43621546, 0.454986431]])
    repeat_number = 10
    repeat_data, beta = generate_repeat_data(repeat_number, setting)  # 重复进行多次单地点实验，其中beta值保持不变，让X和误差项的值发生改变，从而改变y值
    coef_matrix_lasso = np.zeros([repeat_number, 20])  # 用来存贮十次模拟中lasso方法估计系数的情况
    coef_matrix_weight_lasso = np.zeros([repeat_number, 20])  # 用来存贮十次模拟中weight_lasso方法估计系数的情况
    coef_matrix_sim = np.zeros([repeat_number, 20])  # 用来存贮十次模拟中sim方法估计系数的情况
    threshold = 0.00001
    b_min, b_max = bw_selection(g_coords)
    bw = np.linspace(b_min + 0.01, b_max, 10)  # 提前给定的bw的取值
    gwr_bw = np.median(bw)
    lamda = np.linspace(0.01, 100, 100)
    coef_lamd_lasso = np.zeros([len(lamda), 20])  # 用来存储每个lambda下lasso方法系数估计的变化
    coef_lamd_weight_lasso = np.zeros([len(lamda), 20])  # 用来存储每个lambda下weight_lasso方法系数估计的变化
    coef_lamd_sim = np.zeros([len(lamda), 20])  # 用来存储每个lambda下sim方法系数估计的变化
    select_point = 1  # 选择要展示的地点位置
    for i in range(len(lamda)):
        for repeat in range(repeat_number):
            print("lamd=", lamda[i], "bw=", gwr_bw, "repeat_number=", repeat + 1, "calcaulating...")
            try:
                #########################导入数据并进行数据划分###############################
                data_generate = repeat_data[repeat]
                X = np.array(data_generate.iloc[:, 0:p + 2])
                y = np.array(data_generate.iloc[:, p + 2])
                X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X, y, test_size=0.3,
                                                                            random_state=RANDOM_STATE)  # 划分训练集和测试集
                ###################################计算基本信息###################################
                X_train_x, X_test_c, y_train_x, y_test_c = train_test_split(X_train_o, y_train_o, test_size=0.4,
                                                                            random_state=RANDOM_STATE)  # 划分训练集和测试集
                lasso_coef, weight_lasso_coef = [], []
                lasso_intercept, weight_lasso_intercept = [], []
                Aerfa = Kernel(g_coords, gwr_bw, 'bi-square')  # 空间矩阵计算

                ########################Group+GWR#############################
                simultaneous_model = GWGPL(lamd=lamda[i], tol=1e-10, max_iter=100)
                simultaneous_model.fit(X_train_x, y_train_x, Aerfa, g_coords, p, l)
                coef_matrix_sim[repeat, :] = simultaneous_model.coef_.reshape((l,p))[0, :].reshape(1, -1)

                ########################lasso+GWR###############################
                for loc in range(select_point):
                    aerfa = Aerfa.iloc[:, loc]
                    XX_test, yy_test = add_intercept(
                        X_test_o[list((X_test_o[:, 0] == g_coords[loc][0]) & (X_test_o[:, 1] == g_coords[loc][1])), :][
                        :, 2:]), \
                                       y_test_o[list((X_test_o[:, 0] == g_coords[loc][0]) & (
                                                   X_test_o[:, 1] == g_coords[loc][1]))]  # 测试的时候使用该地点的数据进行测试
                    if len(XX_test) == 0:  # 如果该地点没有数据就不再进行预测
                        continue
                    # #######################进行局部加权+Lasso##################################
                    weight_lasso_model = one_point_estimation(add_weight=True, lamd=lamda[i], tol=1e-10, max_iter=100)
                    weight_lasso_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
                    coef_matrix_weight_lasso[repeat, :] = weight_lasso_model.coef_

                ########################lsq+weight+lasso####################################
                for loc in range(select_point):
                    aerfa = Aerfa.iloc[:, loc]
                    XX_test, yy_test = add_intercept(
                        X_test_o[list((X_test_o[:, 0] == g_coords[loc][0]) & (X_test_o[:, 1] == g_coords[loc][1])), :][
                        :, 2:]), \
                                       y_test_o[list((X_test_o[:, 0] == g_coords[loc][0]) & (
                                                   X_test_o[:, 1] == g_coords[loc][1]))]  # 测试的时候使用该地点的数据进行测试
                    if len(XX_test) == 0:  # 如果该地点没有数据就不再进行预测
                        continue
                    ########################Lasso##################################
                    lasso_model = one_point_estimation(add_weight=False,  lamd=lamda[i], tol=1e-10, max_iter=100)
                    lasso_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
                    coef_matrix_lasso[repeat, :] = lasso_model.coef_
            except:
                continue
        print('lamda = ', lamda[i], ', accomplish')
        coef_lamd_lasso[i, :] = coef_matrix_lasso.mean(axis=0)
        coef_lamd_weight_lasso[i, :] = coef_matrix_weight_lasso.mean(axis=0)
        coef_lamd_sim[i, :] = coef_matrix_sim.mean(axis=0)
    print('**********存储最终系数中........************')
    output_dir = 'lasso_path_fixed_location_1/'
    if output_dir[-1] != "/":
        output_dir += "/"
    if not os.path.isdir(output_dir):
        print("Directory doesn't exist, creating it")
        os.mkdir(output_dir)
    pd.DataFrame(coef_lamd_lasso).to_excel(output_dir + "Coef_lasso" + ".xlsx")
    pd.DataFrame(coef_lamd_weight_lasso).to_excel(output_dir + "Coef_weight_lasso" + ".xlsx")
    pd.DataFrame(coef_lamd_sim).to_excel(output_dir + "Coef_sim" + ".xlsx")