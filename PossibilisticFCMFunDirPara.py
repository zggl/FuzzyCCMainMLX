# 2024-11-10
import numpy as np
import pandas as pd
#initialize_membership_matrix

def initialize_membership_matrix(n_samples, n_clusters):
    membership_matrix = np.random.dirichlet(np.ones(n_clusters), size=n_samples)
    return membership_matrix

#calculate_cluster_centers

def calculate_cluster_centers(data, membership_matrix, m):
    num = np.dot((membership_matrix ** m).T, data)
    den = np.sum(membership_matrix ** m, axis=0)[:, np.newaxis]
    centers = num / den
    return centers

# update_membership_matrix

def update_membership_matrix(data, centers, membership_matrix, eta, m, n_clusters):
    dist = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
    dist = np.fmax(dist, np.finfo(np.float64).eps)
    n_samples = data.shape[0]
    for i in range(n_samples):
        for j in range(n_clusters):
            denom = np.sum([(dist[i, j] / dist[i, k]) ** (2 / (m - 1)) \
                                for k in range(n_clusters)])
            membership_matrix[i, j] = 1 / denom

    for i in range(n_samples):
        for j in range(n_clusters):
            t_ij = eta[j] / (eta[j] + dist[i, j])
            membership_matrix[i, j] = (membership_matrix[i, j] ** m) * t_ij

    return membership_matrix

# calculate_eta

def calculate_eta(data, centers, membership_matrix, m):
    n_clusters = centers.shape[0]
    dist = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
    dist = np.fmax(dist, np.finfo(np.float64).eps)

    eta = np.zeros(n_clusters)
    for j in range(n_clusters):
        eta[j] = np.sum((membership_matrix[:, j] ** m) * dist[:, j]) / \
                   np.sum(membership_matrix[:, j] ** m)
    return eta


def calculate_centers(data, U, m):
    np.random.seed(42)
    um = U ** m
    centers = np.dot(um.T, data) / np.sum(um, axis=0, keepdims=True).T
    return centers

def update_membership(data, centers, m):
    np.random.seed(42)
    distance_matrix = np.linalg.norm(data[:, np.newaxis] - centers, axis=2) ** (2/(m-1))
    distance_matrix = np.fmax(distance_matrix, np.finfo(np.float64).eps)
    um = 1.0 / distance_matrix
    um = np.fmax(um, np.finfo(np.float64).eps)
    u_new = um / np.sum(um, axis=1, keepdims=True)
    u_new = np.fmax(u_new, np.finfo(np.float64).eps)
    return u_new

# FCM

def fuzzy_c_means(data, n_clusters, m, max_iters, tol):
    np.random.seed(42)
    n_samples, n_features = data.shape

    
    U = np.random.rand(n_samples, n_clusters)
    U /= np.sum(U, axis=1, keepdims=True)
    
    for iteration in range(max_iters):
        
        U_old = U.copy()

        centers = calculate_centers(data, U, m)
        U = update_membership(data, centers, m)
        diff = np.linalg.norm(U - U_old)
        
        if diff < tol:
            break
    return U, centers

# PFCM

def pfc_means(data, n_clusters, m=2.0, max_iter=100, tol=1e-4):
    n_samples = data.shape[0]
    membership_matrix = initialize_membership_matrix(n_samples, n_clusters)
    centers = calculate_cluster_centers(data, membership_matrix, m)
    eta = calculate_eta(data, centers, membership_matrix, m)

    for iteration in range(max_iter):
        centers_old = centers.copy()
        membership_matrix = update_membership_matrix(data, centers, membership_matrix, 
                                                        eta, m, n_clusters)
        centers = calculate_cluster_centers(data, membership_matrix, m)
        eta = calculate_eta(data, centers, membership_matrix, m)

        if np.linalg.norm(centers - centers_old) < tol:
            break

    return centers, membership_matrix

# minmax(df):
def minmax(df):
    columns = df.columns[:]
    for column in columns:
        max = df[column].max()
        min = df[column].min()
        df[column] = (df[column] - min)/(max - min)
    return df.to_numpy()

#----------------------------------------------------------------------------
# Example usage
# Example data with 100 samples and 2 features

#data = np.random.rand(100, 2) 
#n_clusters = 3

df = pd.read_csv(file, header=None)
# Select the first four columns
data = df.iloc[:, :4]
data_labels = df.iloc[:, :5]
# Display the result
print(data.shape)

# nornalzation
norm_df = minmax(data)
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
#print(Num_Cluter)
cluster_number=int(Num_Cluter)
print('-------------------------------------cluster_number: ')
print(cluster_number)

centers, membership_matrix = pfc_means(data_train, cluster_number)
PFCM_U=membership_matrix.T
print("Cluster Centers:\n", centers)
print("Membership Matrix:\n", membership_matrix)
