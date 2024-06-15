from numpy.linalg import pinv  # 矩阵求逆
from numpy import dot  # 矩阵点乘
from numpy import mat  # 二维矩阵

X = mat([2, 1, 2, 2, 1, 3, 2, 1, 4]).reshape(3, 3)  # x为1,2,3
Y = mat([5, 10, 15]).reshape(3, 1)  # y为5,10,15
a = dot(dot(pinv(dot(X.T, X)), X.T), Y)  # 最小二乘法公式
print(a)