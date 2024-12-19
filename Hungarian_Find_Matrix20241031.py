import numpy as np
from scipy.optimize import linear_sum_assignment
import math
import latexify
# Alg: \href{http://dx.doi.org/10.1109/TAES.2016.140952}
# {On implementing 2D rectangular assignment algorithms},
#{IEEE Transactions on Aerospace and Electronic Systems}.

# 计算成本矩阵，这里使用欧氏距离
def Cost_MatrixCal(A,B):
    cost_matrix = np.zeros((A.shape[0], B.shape[0]))
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            cost_matrix[i][j] = np.linalg.norm(A[i] - B[j])
    return cost_matrix

def Assignment_ModifiedJonkerVolgenantAlg(cost_matrix):
    # 应用匈牙利算法
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    return row_ind, col_ind

#print(A)
#print(B)
cost_matrix=Cost_MatrixCal(A,B)

#print(cost_matrix)

row_ind, col_ind=Assignment_ModifiedJonkerVolgenantAlg(cost_matrix)
# 输出索引
#print(row_ind)
#print(col_ind)
'''
# 输出最佳匹配结果
print("匹配结果:")
for r, c in zip(row_ind, col_ind):
    print(f"A 中的行 {r} 与 B 中的行 {c} 匹配，成本为 {cost_matrix[r, c]}")

# 使用匹配索引重新排列矩阵 A 和 B

A_matched = A[row_ind]
B_matched = B[col_ind]

print("重新排序后的矩阵 A:")
print(A_matched)

print("与之匹配的矩阵 B:")
print(B_matched)
'''