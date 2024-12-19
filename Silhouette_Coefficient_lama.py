import numpy as np
import pandas as pd

# jarak antara dua data
def get_Distance(x, y):
    return np.around(np.sum((x - y) ** 2), decimals=3)

# rata-rata jarak antar data di cluster yang sama
def get_a(x, cluster_x):
    a = 0
    if len(cluster_x) > 1:
        for i in range(len(cluster_x)):
            if  x != i:
                a += get_Distance(cluster_x[x], cluster_x[i])
        a /= (len(cluster_x)-1)
    return np.around(a, decimals=3)

# rata-rata jarak antar data di cluster berbeda
def get_b(x, indeks_cluster_x, clusters):
    b = 0; n = 0
    for j in range(len(clusters)):
        if j != indeks_cluster_x:
            n += len(clusters[j])
            b += np.sum([get_Distance(clusters[indeks_cluster_x][x], y) for y in clusters[j]])
    b /= n
    return np.around(b, decimals=3)

def get_Silhouette(cluster):
    # preparation from df
    n_cluster = cluster.Cluster.value_counts().index.tolist()
    local_S = [0]
    global_S = 0
    
    if len(n_cluster) >= 2:
        all_cluster = []
        for i in n_cluster:
            df_i = cluster.loc[cluster.Cluster == i]
            df_i = df_i.drop(['Data', 'Cluster'], axis=1)
            arr_i = df_i.to_numpy()
            all_cluster.append(arr_i)

        # get silhouette
        global_S = 0
        global_n = 0
        local_S = []

        for i in range(len(n_cluster)):
            l_s = 0
            for x in range(len(all_cluster[i])):
                a = get_a(x, all_cluster[i])
                b = get_b(x, i, all_cluster)
                s = (b - a) / max(a, b)
                l_s += s

            global_n += len(all_cluster[i])
            global_S += l_s

            l_s /= len(all_cluster[i])
            local_S.append(l_s)

        global_S /= global_n

    # local silhouette to df
    local_S_df = pd.DataFrame(list(zip(n_cluster, local_S)), columns=['Cluster', 'silhouette']).sort_values(by='Cluster').reset_index(drop=True)

    return global_S, local_S_df
