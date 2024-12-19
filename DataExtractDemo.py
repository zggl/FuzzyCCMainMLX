import numpy as np
# 示例1: 获取一维数组的一部分数据
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
start = 2
end = 6
result = data[start:end] # 获取从第2行到第6行的数据
print(result) # 输出: [3, 4, 5, 6, 7]
print('-------------------------------------')
# 示例2: 获取二维数组的一部分数据
# 创建示例 ndarray
array = np.array([[1, 2, 3], 
                  [4, 5, 6], 
                  [7, 8, 9], 
                  [10, 11, 12], 
                  [13, 14, 15], 
                  [16, 17, 18]])

# 提取第 1, 3, 5 行（索引从 0 开始，所以选择 [0, 2, 4] 行）
selected_rows1 = array[[0, 2, 4], :]

# 打印结果
print("Selected rows:")
print(selected_rows1)

# 提取第 1:2（即第 1 和第 2 行）和第 4:5（即第 4 和第 5 行）
selected_rows2 = np.vstack((array[0:2, :], array[3:5, :]))
# 打印结果
print("Selected rows:")
print(selected_rows2)


import numpy as np

# 创建示例 ndarray
array = np.arange(200).reshape(200, 1)  # 假设有一个 200 行的数组

# 提取第 1:5 行（索引 0:5）、51:55 行（索引 50:55）和 101:105 行（索引 100:105）
selected_rows = np.vstack((array[0:5, :], array[50:55, :], array[100:105, :]))

# 打印结果
print("Selected rows:")
print(selected_rows)

import pandas as pd

# 读取 IRIS 数据集（通常可以从 CSV 文件或 sklearn 数据集中读取）
# 使用 sklearn 内置数据集示例
from sklearn.datasets import load_iris

# 加载数据集并转换为 DataFrame
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# 提取第 1:5 行（索引 0:5）、51:55 行（索引 50:55）、101:105 行（索引 100:105）
selected_rows = pd.concat([iris_df.iloc[0:5], iris_df.iloc[50:55], iris_df.iloc[100:105]])

# 打印结果
print("Selected rows:")
print(selected_rows)