# tested 2024-11-10
import numpy as np
import pandas as pd
from datetime import datetime as time
import Silhouette_Coefficient as sc

print("\033c", end="")
eps = np.finfo(float).eps

df = pd.read_csv("./datasets/Fish.csv")
start = time.now()
df = df.sample(frac=1).reset_index(drop=True)
# df
df_copy = df.copy()
df_copy.pop('No')
df_copy.pop('Species')
df_copy
print('------------------------------------df_copy type')
print(type(df_copy.shape))
print('------------------------------------df_copy.shape')
print(df_copy.shape)
# minmax(df):

def minmax(df):
    columns = df.columns[:]
    for column in columns:
        max = df[column].max()
        min = df[column].min()
        df[column] = (df[column] - min)/(max - min)
    return df.to_numpy()

norm_df = minmax(df_copy)
norm_df
proportion = int(len(norm_df)*0.7)             # 70% dataset
data_train = norm_df[:proportion]
data_test = norm_df[proportion:]
print('------------------------------------Training/test data size: ')
print('Training data:', len(data_train),'\n Test data:',len(data_test))

# set_V_awal(x, u, c, w):

def set_V_awal(x, u, c, w):
    v = np.zeros([c, x.shape[1]]) #c x len(x)
    '''
    print('-----------------------------x.shape')
    print(x.shape)# len(x) x c
    print('------------------------------len(x)')
    print(len(x))
    print('-----------------------------v.shape')
    print(v.shape)
    print('------------------------------len(v)')
    print(len(v))
    print('-----------------------------v.shape')
    print(v.shape)
    print('-----------------------------u.shape')
    print(u.shape)
    print('-----------------------------w')
    print(w)
    print('------------------------------len(u)')
    print(len(u))# len(x) x c
    '''
    for k in range(len(v)):
        for j in range(len(v[k])):
            v[k][j] = ( sum( ( (u[i][k]**w) * x[i][j]) for i in range(len(x)) ) ) /\
                                 ( sum( (u[i][k]**w ) for i in range(len(u)))+eps)

    return v

def update_t(t, x, v, n):
    for i in range(len(t)):
        for k in range(len(t[i])):
            t[i][k] = ((sum((x[i]-v[k])**2)+eps)**((-1)/(n-1))) \
                    / (sum(((sum((x[i]-v[k])**2)+eps)**((-1)/(n-1))+2*eps)\
                         for i in range(len(x))))

    return t

def update_V(v, x, u, t, w, n):
    #print(len(x))
    for k in range(len(v)):
        for j in range(len(v[k])):
            v[k][j] = (sum((((u[i][k]**w) + (t[i][k]**n)) * \
                        x[i][j]) for i in range(len(x)))) \
                    / (sum(((u[i][k]**w) + (t[i][k]**n)) for i in range(len(x)))+eps)

    return v

def update_U(u, x, v, w):
    for i in range(len(u)):
        for k in range(len(u[i])):
            u[i][k] = ((sum((x[i]-v[k])**2)+eps)**((-1)/(w-1))) \
                    / ( sum(((sum((x[i]-v[k])**2)+eps)**((-1)/(w-1)) ) 
                    for i in range(len(x)))+eps)

    return u

def get_P1(x, v, u, t, w, n):
    return sum((sum(((sum((x[i]-v[k])**2))) * ((u[i][k]**w) + (t[i][k]**n)) \
           for k in range(len(v)))) for i in range(len(x)))

# 

def FPCM(x, c, w, n, max_i, P0, e):
    list_i = []
    list_P0 = []
    list_P1 = []
    list_abs_P1_min_P0 = []
    
    U = np.random.rand(len(x), c)
    V = set_V_awal(x=x, u=U, c=c, w=w)
    t = np.zeros(U.shape)
    t = update_t(t=t, x=x, v=V, n=n)

    i = 1
    while(i <= max_i):
        P1 = get_P1(x=x, v=V, u=U, t=t, w=w, n=n)
        abs_P1_min_P0 = abs(P1 - P0)

        list_i.append(i)
        list_P0.append(P0)
        list_P1.append(P1)
        list_abs_P1_min_P0.append(abs_P1_min_P0)

        V = update_V(v=V, x=x, u=U, t=t, w=w, n=n)
        U = update_U(u=U, x=x, v=V, w=w)
        t = update_t(t=t, x=x, v=V, n=n)

        if abs_P1_min_P0 < e:
            break
        else:
            P0 = P1
            i += 1

    p_df = pd.DataFrame(list(zip(list_i, list_P1, list_P0, list_abs_P1_min_P0)), 
                                    columns=['iteration', 'P1', 'P0', '|P1-P0|'])
    return V,p_df

# get_cluster(x, v):

def get_cluster(x, v):
    data = [(i+1) for i in range(len(x))]
    fitur = [('F' + str(m+1)) for m in range(len(x[0]))]
    cluster = [(np.argmin([np.sqrt(sum((xi-vi)**2)) for vi in v]) + 1) for xi in x]

    cluster_df = pd.DataFrame(x, columns=fitur)
    cluster_df['Cluster'] = cluster
    cluster_df.insert(0, 'Data', data)
    return cluster_df

V_train, P_df = FPCM(x=data_train, c=2, w=2, n=2, max_i=100, P0=0, e=0.00001)
# pusat cluster (1 data pertama)
V_train[:1]
V_train.shape
# fungsi objektif setiap iterasi
P_df
cluster_df = get_cluster(x=data_test, v=V_train)
# hasil clustering (5 data teratas)
cluster_df.head()
cluster_df.groupby(['Cluster'])['Data'].count().reset_index().rename(columns\
                            ={"Data" : "Count (Data)"})
# clustering data format *.csv
cluster_df.to_csv('FPCM_clustering_result.csv', index=False)
# FPCM_clustering_result.csv
# pd.read_csv('hasil_clustering_FPCM.csv').head()
global_s, local_s_df = sc.get_Silhouette(cluster_df)
# mark silhouette pada setup cluster
local_s_df
print('------------------------------------Silhouette Coefficient index')
# system clustering
print('Silhouette Coefficient index:',global_s)
t = time.now() - start
print('Elapse time:',str(t.seconds)+","+str(t.microseconds)[:3],'second')

