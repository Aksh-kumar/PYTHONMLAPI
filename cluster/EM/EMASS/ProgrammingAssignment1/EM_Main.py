#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt 
import imageio
import os
from scipy.stats import multivariate_normal
import pandas as pd
from functools import reduce


# In[2]:


# compute ln(exp^e1+expe2....)
def ln_sum_exp(Z):
    return np.log(np.sum(np.exp(Z)))

def get_log_likelihood(data, weights, means, covs):
    n_clusters = len(means)
    dimension = len(data[0]) 
    sum_ln_exp = 0
    for d in data:
        Z = np.zeros(n_clusters)
        for k in range(n_clusters):
            # compute exponential term in multivariate_normal
            delta = np.array(d) - means[k]
            exponent_term = np.dot(delta.T, np.dot(np.linalg.inv(covs[k]), delta))
            # Compute loglikelihood contribution for this data point and this cluster
            Z[k] += np.log(weights[k])
            Z[k] -= 1/2. * (dimension * np.log(2*np.pi) + np.log(np.linalg.det(covs[k])) + exponent_term)
        # Increment loglikelihood contribution of this data point across all clusters
        sum_ln_exp += ln_sum_exp(Z)
    return sum_ln_exp


# In[3]:


# E step - get the responsibility if cluster parameter is given
def get_responsibilities(data, weights, means, covariances):
    n_data = len(data)
    n_clusters = len(means)
    resp = np.zeros((n_data, n_clusters))
    for i in range(n_data):
        for k in range(n_clusters):
            resp[i, k] = weights[k]* multivariate_normal.pdf(data[i],means[k],covariances[k])
    # Add up responsibilities over each data point and normalize
    row_sums = resp.sum(axis=1)[:, np.newaxis]
    resp = resp / row_sums
    return resp


# In[4]:


#M-step update the parameter if given cluster responsibilty is given
# get sum of all responsibilty for a particular cluster
# resp = nxk metrix i.e resp[number_of_label, nuber_of_cluster]
get_soft_counts = lambda resp : np.sum(resp, axis=0)


# In[5]:


# get list of weights which represents how much each cluster represented over all data points
# parameter count list of sum of soft counts for all clusters
def get_weights(counts):
    n_clusters = len(counts)
    sum_count = np.sum(counts)
    weights = np.array(list(map(lambda k : counts[k]/sum_count, range(n_clusters))))
    return weights


# In[6]:


def get_means(data, resp, counts):
    n_clusters = len(counts)
    n_data = len(data)
    means = np.zeros((n_clusters, len(data[0])))
    for k in range(n_clusters):
        weighted_sum = reduce(lambda x,i : x + resp[i,k]*data[i], range(n_data), 0)
        means[k] = weighted_sum/counts[k]
    return means


# In[7]:


data_tmp = np.array([[1.,2.],[-1.,-2.]])
resp = get_responsibilities(data=data_tmp, weights=np.array([0.3, 0.7]),
                                means=[np.array([0.,0.]), np.array([1.,1.])],
                                covariances=[np.array([[1.5, 0.],[0.,2.5]]), np.array([[1.,1.],[1.,2.]])])
print(resp)
counts = get_soft_counts(resp)
means = get_means(data_tmp, resp, counts)

if np.allclose(means, np.array([[-0.6310085, -1.262017], [0.25140299, 0.50280599]])):
    print('Checkpoint passed!')
else:
    print('Check your code again.')


# In[8]:


# update covariances metrix from responsibity metrix, means and counts
def get_covariances(data, resp, counts, means):
    n_clusters = len(counts)
    dimension = len(data[0])
    n_data = len(data)
    covariances = np.zeros((n_clusters, dimension, dimension))
    for k in range(n_clusters):
        weighted_sum = reduce(lambda x, i :x + resp[i,k] * np.outer((data[i]-means[k]), (data[i]-means[k]).T),
                              range(n_data), np.zeros((dimension, dimension)))
        covariances[k] = weighted_sum/counts[k]
    return covariances


# In[9]:


data_tmp = np.array([[1.,2.],[-1.,-2.]])
resp = get_responsibilities(data=data_tmp, weights=np.array([0.3, 0.7]),
                                means=[np.array([0.,0.]), np.array([1.,1.])],
                                covariances=[np.array([[1.5, 0.],[0.,2.5]]), np.array([[1.,1.],[1.,2.]])])
counts = get_soft_counts(resp)
means = get_means(data_tmp, resp, counts)
covariances = get_covariances(data_tmp, resp, counts, means)

if np.allclose(covariances[0], np.array([[0.60182827, 1.20365655], [1.20365655, 2.4073131]])) and     np.allclose(covariances[1], np.array([[ 0.93679654, 1.87359307], [1.87359307, 3.74718614]])):
    print('Checkpoint passed!')
else:
    print('Check your code again.')


# In[10]:


# EM algorithm
def EM(data, init_means, init_covariances, init_weights, maxiter=1000, thresh=1e-4, verbose=True):
    # Make copies of initial parameters
    means = init_means[:]
    covariances = init_covariances[:]
    weights = init_weights[:]
    # Infer dimensions of dataset and the number of clusters
    n_data = len(data)
    dimension = len(data[0])
    n_clusters = len(means)
    # Initialize some useful variables
    resp = np.zeros((n_data, n_clusters))
    ll = get_log_likelihood(data, weights, means, covariances)
    ll_list = [ll]
    for it in range(maxiter):
        if verbose and it % 5 == 0:
            print("Iteration %s" % it)
        # E-step: calculate responsibilities
        resp = get_responsibilities(data, weights, means, covariances)
        # M-step calculate cluster parameter
        counts = get_soft_counts(resp)
        weights = get_weights(counts)
        means = get_means(data, resp, counts)
        covariances = get_covariances(data, resp, counts, means)
        ll_new = get_log_likelihood(data, weights, means, covariances)
        ll_list.append(ll_new)
        if (ll_new - ll) < thresh and ll_new > -np.inf:
            break
        ll = ll_new
    out = {'weights': weights, 'means': means, 'covariances': covariances, 'loglikelihood': ll_list, 'responsibility': resp}
    return out


# In[11]:


def EM_initializer_from_responsibility(data, resp) :
    counts = get_soft_counts(resp)
    weights = get_weights(counts)
    means = get_means(data, resp, counts)
    covariances = get_covariances(data, resp, counts, means)
    return mean, covariancesriances, weights


# In[12]:


def pairwse_euclidean_dis(data, centroids) :
    assert data.shape[1] == centroids.shape[1]
    dis_list = list(map(lambda centroid : (np.sum((data - centroid)**2, axis=1)**0.5)[:, np.newaxis], centroids))
    return np.concatenate(dis_list, axis=1)
# End
a=np.array([[1,2],[3,4]])
c=np.array([[1,1],[0,0],[5,6]])
print(pairwse_euclidean_dis(a,c))


# In[13]:


def assign_clusters(data, centroids):
    distances_from_centroids = pairwse_euclidean_dis(data, centroids)
    cluster_assignment = np.argmin(distances_from_centroids,axis=1)
    return cluster_assignment


# In[14]:


data = np.array([[1., 2., 0.],
                 [0., 0., 0.],
                 [2., 2., 0.]])
centroids = np.array([[0.5, 0.5, 0.],
                      [0., -0.5, 0.]])
cluster_assignment = assign_clusters(data, centroids)
print(cluster_assignment)


# In[151]:


def update_centroids(data, k, cluster_assignment):
    new_centroids = []
    for i in range(k):
        cluster_points = data[cluster_assignment==i]
        centroid = cluster_points.mean(axis=0)
        centroid = centroid.ravel()
        new_centroids.append(centroid)
    return np.array(new_centroids)


# In[149]:


def get_heterogeneity(data, k, centroids, cluster_assignment):
    heterogeneity = 0.0
    for i in range(k):
        cluster_points = data[cluster_assignment==i, :]
        if cluster_points.shape[0] > 0: # check if i-th cluster is non-empty
            distances = pairwse_euclidean_dis(cluster_points, np.array([centroids[i]]))
            squared_distances = distances**2
            heterogeneity += np.sum(squared_distances)
    return heterogeneity


# In[17]:


def k_meanpp_initialize(data, k, seed=None):
    if seed is not None: # useful for obtaining consistent results
        np.random.seed(seed)
    dimension = data.shape[1]
    centroids = np.zeros((k, dimension))
    init_cen = np.random.randint(data.shape[0])
    centroids[0] = data[init_cen]
    #need to flatten() because it will remove extra dimension returned by pairwise distances used later in devide
    squared_distances = pairwse_euclidean_dis(data, centroids[0:1]).flatten()**2
    for i in range(1, k):
        # np.random.choice(range of numbers, size, probablity) pick the number from range of number whose assiciate
        # probablity is higher given in p list
        id_centroids = np.random.choice(data.shape[0], 1, p=squared_distances/sum(squared_distances))
        centroids[i] = data[id_centroids]
        squared_distances = np.min(pairwse_euclidean_dis(data, centroids[0:i+1])**2,axis=1)
    return centroids


# In[18]:


def kmeans(data, k, initial_centroids, maxiter=1000):
    centroids = initial_centroids[:]
    prev_cluster_assignment = None 
    for itr in range(maxiter):        
        cluster_assignment = assign_clusters(data, centroids)
        centroids = update_centroids(data, k, cluster_assignment)
        # if no changes detected
        if prev_cluster_assignment is not None and (prev_cluster_assignment==cluster_assignment).all():
            break
        prev_cluster_assignment = cluster_assignment[:]
    return centroids, cluster_assignment


# In[19]:


data = np.array([[1., 2., 0.],
                 [0., 0., 0.],
                 [2., 2., 0.]])
centroids = np.array([[0.5, 0.5, 0.],
                      [0., -0.5, 0.]])
cluster_assignment = assign_clusters(data, centroids)
print(cluster_assignment)
k=2
print(update_centroids(data, k, cluster_assignment))


# In[121]:


def get_image_feature(path) :
    try :
        Im = imageio.imread(os.path.join(path), pilmode='RGB')
        temp = Im/255. # removing opacity value and divide by 255 to get in fraction
        mn = temp.sum(axis=0).sum(axis=0)/(temp.shape[0]*temp.shape[1])
        return mn
    except :
        return None


# In[141]:


def get_file_list(path, image_EXT, df=None) :
    dataset = df
    for files in os.listdir(os.path.join(path)) :
        f_path = os.path.join(path, files)
        if os.path.isfile(f_path) :
            temp = os.path.splitext(files)
            ext = temp[1]
            if str.lower(ext) in image_EXT :
                f_name = temp[0]
                mn = get_image_feature(f_path)
                #     file_name, path,extension,red, green, blue
                data = [f_name, f_path, ext, mn[0], mn[1], mn[2]]
                data_dic = dict(zip(list(df.columns), data))
                if df is not None :
                    dataset = dataset.append(data_dic, ignore_index=True)
        if os.path.isdir(f_path) :
            dataset = get_file_list(f_path, image_EXT, dataset)
    return dataset
# End


# In[142]:


#path = '..\..\..\..\Temp\Demo_Pic.png'
path = r'.\images'
#path = '.\images\cloudy_sky\ANd9GcQ-HRsM6J42Mc2BJaSFXbxIidQEep8bjhaHdn-xUCfw91f0MHtE.jpg'
dataset = pd.DataFrame([],columns=['Image_Name', 'Path', 'Extension', 'R', 'G', 'B' ])
image_EXT = ['.png', '.jpeg', '.jpg', '.jif', '.jpe']
dataset = get_file_list(os.path.join(path), image_EXT, dataset)
dataset


# In[145]:


X = dataset.iloc[:, [3, 4, 5]].values


# In[150]:


max_k = 100
heterogeneity = []
k_list = []
for k in range(1,max_k) :
    init_cent = k_meanpp_initialize(X, k, seed=0)
    centroids, cluster_assignment =  kmeans(X, k, init_cent)
    heterogeneity.append(get_heterogeneity(X, k, centroids, cluster_assignment))
    k_list.append(k)


# In[20]:


# test 
def generate_MoG_data(num_data, means, covariances, weights):
    """ Creates a list of data points """
    num_clusters = len(weights)
    data = []
    for i in range(num_data):
        #  Use np.random.choice and weights to pick a cluster id greater than or equal to 0 and less than num_clusters.
        k = np.random.choice(len(weights), 1, p=weights)[0]

        # Use np.random.multivariate_normal to create data from this cluster
        x = np.random.multivariate_normal(means[k], covariances[k])

        data.append(x)
    return data
# Model parameters
init_means = [
    [5, 0], # mean of cluster 1
    [1, 1], # mean of cluster 2
    [0, 5]  # mean of cluster 3
]
init_covariances = [
    [[.5, 0.], [0, .5]], # covariance of cluster 1
    [[.92, .38], [.38, .91]], # covariance of cluster 2
    [[.5, 0.], [0, .5]]  # covariance of cluster 3
]
init_weights = [1/4., 1/2., 1/4.]  # weights of each cluster

# Generate data
np.random.seed(4)
data = generate_MoG_data(100, init_means, init_covariances, init_weights)


# In[21]:


plt.figure()
d = np.vstack(data)
plt.plot(d[:,0], d[:,1],'ko')
plt.rcParams.update({'font.size':16})
plt.tight_layout()


# In[22]:


np.random.seed(4)

# Initialization of parameters
chosen = np.random.choice(len(data), 3, replace=False)
initial_means = [data[x] for x in chosen]
initial_covs = [np.cov(data, rowvar=0)] * 3
initial_weights = [1/3.] * 3

# Run EM 
results = EM(data, initial_means, initial_covs, initial_weights)


# In[23]:


results


# In[ ]:





# In[ ]:




