import numpy as np
from scipy.stats import pearsonr

np.random.seed(0)
size = 300
x = np.random.normal(0, 1, size)
print("Lower noise：", pearsonr(x, x + np.random.normal(0, 1, size)))
print("Higher noise：", pearsonr(x, x + np.random.normal(0, 10, size)))


from sklearn.feature_selection import SelectKBest
# 选择K个最好的特征，返回选择特征后的数据
# 第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
# 参数k为选择的特征个数
SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)


