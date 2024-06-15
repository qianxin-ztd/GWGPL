import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

def select_data(data):
    data1 = data.groupby(["longitude", "latitude"]).agg("count")
    # 获取纬度
    coord = np.array(data1.index)
    count = np.array(data1.iloc[:, 0])
    select = []
    for i in range(len(count)):
        if count[i] <= N:
            select.append(False)
        else:
            select.append(True)
    data2 = np.array([[coord[i][0], coord[i][1], select[i]] for i in range(data1.shape[0])])
    df = data
    for i in range(len(data)):
        for j in range(len(data2)):
            if data.iloc[i, 1] == data2[j, 0] and data.iloc[i, 2] == data2[j, 1] and data2[j, 2] == 0:
                df = df.drop(index=i)
    return df

if __name__ == '__main__':
    data = pd.read_excel('./data/data_original1.xlsx')
    N = 30  # 多少以下的地点不再进行考虑
    #################对有序变量和0-1变量进行一些处理###################
    for i in range(len(data.iloc[:, 4])):
        if data.iloc[i, 3] == 2: data.iloc[i, 3] = 0  # 1男0女
        if data.iloc[i, 5] >= 4:  data.iloc[i, 5] = 4 # 将教育变量做一定的整理
        if data.iloc[i, 6] == 3:  data.iloc[i, 6] = 2  # 无业、农、半工半农、非农（1,2,3,4）
        elif data.iloc[i, 6] == 4:  data.iloc[i, 6] = 3
        elif data.iloc[i, 6] == 5:  data.iloc[i, 6] = 4
        else:  data.iloc[i, 6] = 1
        if data.iloc[i, 8] == 2:  data.iloc[i, 8] = 1   # 婚姻变量分为1已婚和0其他情况
        else:  data.iloc[i, 8] = 0
        if data.iloc[i, 9] != 1:  # 非农业与农业（0,1）
            data.iloc[i, 9] = 0
        if data.iloc[i, 10] != 1:  # 非本村与本村（1,2）
            data.iloc[i, 10] = 0
        if data.iloc[i, 17] == 2:  data.iloc[i, 17] = 0  # 将是否有慢性病转为0-1变量
        if data.iloc[i, 19] == 2:  data.iloc[i, 19] = 0  # 将是否使用保险转为0-1变量
        if data.iloc[i, 20] == 2:  data.iloc[i, 20] = 0  # 将是否看过门诊转为0-1变量
        if data.iloc[i, 21] == 2:  data.iloc[i, 21] = 0  # 将是否自我治疗转为0-1变量
    data.iloc[:, 15] = np.log(data.iloc[:, 15] + 1)  # 医院距离偏态对数化处理
    data.iloc[:, 25] = np.log(data.iloc[:, 25] + 1)  # 住院医疗偏态对数化处理
    data.to_excel('aa.xlsx')
    #######################分类变量处理###################################
    before, after = data.iloc[:, :5], data.iloc[:, 8:]
    edu = pd.get_dummies(data.iloc[:, 5])  # 教育变量哑变量处理
    work = pd.get_dummies(data.iloc[:, 6])  # 就业情况哑变量处理,无业、农、半工半农、非农
    institution = pd.get_dummies(data.iloc[:, 7])  # 医疗机构哑变量处理
    edu.columns = ['教育（不识字）', '教育（小学）', '教育（初中）', '教育（高中及以上）']
    work.columns = ['无业', '农业', '半工半农', '非农业']
    institution.columns = ['一级医院', '二级医院', '三级医院', '私立医院', '其他']
    edu = edu.drop('教育（不识字）', axis=1)  # 将高学历作为对照组
    work = work.drop('无业', axis=1)  # 将无业作为对照组
    institution = institution.drop('一级医院', axis=1)  # 将一级医院对照组设置
    data = pd.concat([before, edu, work, institution, after], axis=1)
    data_select = select_data(data)
    data_select.to_excel('./data/data_select1'+'.xlsx')  # 选出符合每个地点样本个数要求的数据