# 2024-11-10
import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.datasets as ds
import skfuzzy as fuzz
from plot import plot
import cmeans
#import os
#os.system('cls' if os.name == 'nt' else 'clear')
###------------------------------------------------------------------------
# minmax(df):
def minmax(df):
    columns = df.columns[:]
    for column in columns:
        max = df[column].max()
        min = df[column].min()
        df[column] = (df[column] - min)/(max - min)
    return df.to_numpy()

###------------------------------------------------------------------------
def generate_data(num_samples, num_features, c, shuffle=False):
    # scikit-learn 1.4.2
    # sklearn.datasets.make_blobs(n_samples=100, n_features=2, *, centers=None, 
    # cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None, 
    # return_centers=False)
    x = ds.make_blobs(n_samples=num_samples, n_features=num_features, centers=c, 
                 cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=False)[0]
    x = x.T
    labels = np.zeros(num_samples)

    j = int(num_samples / c)

    for i in range(c):
        p = i * j
        q = (i + 1) * j
        print()
        labels[p:q] = i

    return x, labels

###------------------------------------------------------------------------
def verify_clusters(x, c, v, u, labels):
    ssd_actual = 0

    for i in range(c):
        # All points in class
        x1 = x[labels == i]
        # Mean of class
        m = np.mean(x1, axis=0)

        for pt in x1:
            ssd_actual += np.linalg.norm(pt - m)

    clm = np.argmax(u, axis=0)
    ssd_clusters = 0

    for i in range(c):
        # Points clustered in a class
        x2 = x[clm == i]

        for pt in x2:
            ssd_clusters += np.linalg.norm(pt - v[i])
    print(ssd_clusters / ssd_actual)
# # N
# num_samples = 3000
# # S
# num_features = 2
# c = 3
# fuzzifier = 1.2
# error = 1E-3
# maxiter = 100
# 
# # np.random.seed(100)
# 
# x, labels = generate_data(num_samples, num_features, c, shuffle=False)
# print(x)

df = pd.read_csv(file, header=None)
# Select the first four columns
df_first_four_columns = df.iloc[:, :4]
print('-------------------------------------df_first_four_columns type: ')
print(type(df_first_four_columns))
df_last_columns = df.iloc[:, :5]
print('-------------------------------------df_last_columnss type: ')
print(type(df_last_columns))

# Display the result
print(df_first_four_columns.shape)
norm_df = minmax(df_first_four_columns)
#norm_df
print('-------------------------------------norm_df size: ')
print(norm_df.shape)
#proportion = int(len(norm_df)*0.7)             # 70% dataset
proportion = int(len(norm_df)*1) 
print('-------------------------------------proportion size: ')
print(proportion)
data_train = norm_df[:proportion]
data_test = norm_df[proportion:]
print('-------------------------------------Training/test data size: ')
print('Training data:', len(data_train),'\n Test data:',len(data_test))


# cmeans.pcm

digits = df_first_four_columns.values.T
labels = df_last_columns.values.T
#digits = np.array(digits.data).T
cluster_number=int(Num_Cluter)

print('-------------------------------------cluster_number: ')
print(cluster_number)

fuzzifier = 1.2
error = 1E-3
maxiter = 1000

v, v0, u, u0, d, t = cmeans.pcm(digits, cluster_number, fuzzifier, error, maxiter)

'''
v, v0, u, u0, d, t = cmeans.fcm(x, c, fuzzifier, error, maxiter)
plot(x.T, v, u, c)

print("Blobs")
verify_clusters(x.T, c, v, u, labels)

iris = ds.load_iris()
labels = iris.target_names
target = iris.target
iris = np.array(iris.data).T
c = 3
v, v0, u, u0, d, t = cmeans.fcm(iris, c, fuzzifier, error, maxiter)
iris = iris.T
print("Iris")

verify_clusters(iris, c, v, u, target)
digits = ds.load_digits()
labels = digits.target
digits = np.array(digits.data).T
c = 10
v, v0, u, u0, d, t = cmeans.pcm(digits, c, fuzzifier, error, maxiter)

print("Digits")
verify_clusters(digits.T, c, v, u, labels)
plot(digits.T, v, u, c)
'''
