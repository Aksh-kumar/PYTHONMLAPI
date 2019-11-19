#!/usr/bin/env python
# coding: utf-8

import numpy as np

class KM:
    """
    Low level K mean algorithm implementation
    """
    
    def __init__(self):
        pass


    @staticmethod
    def pairwse_euclidean_dis(data, centroids):
        """ calculate pairwise euclidean distance and return pairwise distance np array
            
            args :
                data (numpy array) : data for calculation of euclidean
                centroids (numpy array) : centroids co-ordinates of clusters
            
            Returns :
                numpy array of distances of points from each clusters
        """

        assert data.shape[1] == centroids.shape[1] # check for equal dimention
        dis_list = list(map(lambda centroid : (np.sum((data - centroid)**2, axis=1)**0.5)[:, np.newaxis], centroids))
        return np.concatenate(dis_list, axis=1)
    

    def _assign_clusters(self, data, centroids):
        """ calculate euclidean distance and assign to the nearest cluster
            
            args :
                data (numpy array) : data points numpy array
                centroids (numpy array) : numpy array of clusters centroids
            
            Returns :
                numpy array of clusters assignments
        """

        distances_from_centroids = self.pairwse_euclidean_dis(data, centroids)
        cluster_assignment = np.argmin(distances_from_centroids,axis=1)
        return cluster_assignment
    

    def _update_centroids(self, data, k, cluster_assignment):
        """ calculate the centroids and update the previous one
            
            args :
                data (numpy array) :  numpy array of observation points
                k (int) : number of clusters
                cluster_assignment (numpy array) :  assignment clusters number array
            
            Returns :
                numpy array of new centroids for clusters
        """

        new_centroids = []
        for i in range(k):
            cluster_points = data[cluster_assignment==i]
            centroid = cluster_points.mean(axis=0)
            centroid = centroid.ravel()
            new_centroids.append(centroid)
        return np.array(new_centroids)
    

    def get_heterogeneity(self, data, k, centroids, cluster_assignment):
        """ calculate heterogeneity to check skewness of assigned cluster
            
            args :
                data (numpy array) : numpy array of oservation points
                k (int) : number of clusters
                centroids (numpy array) : centroids points of each clusters
                cluster_assignment (numpy array) : assignment cluster number for each observation array
            
            Returns :
                float heterogeneity for particular k
        """

        heterogeneity = 0.0
        for i in range(k): # for each cluster calculate skewness
            cluster_points = data[cluster_assignment==i, :]
            if cluster_points.shape[0] > 0: # check if i-th cluster is non-empty
                distances = self.pairwse_euclidean_dis(cluster_points, np.array([centroids[i]]))
                squared_distances = distances**2
                heterogeneity += np.sum(squared_distances)
        return heterogeneity
    

    def _k_meanpp_initialize(self, data, k, seed=None):
        """ K mean plus plus initialization algorithm assigned cluster centroids
            as far as possible to the all cluster which is already assigned

            args :
                data (numpy array) : numpy array of observation points
                k (int) : number of clusters
                seed (int) : set seed for constant randomness for initialization (optional)
            
            Returns :
                numpy array of centroids initialized by k mean ++ algorithm
        """

        if seed is not None: # useful for obtaining consistent results
            np.random.seed(seed)
        dimension = data.shape[1]
        centroids = np.zeros((k, dimension))
        init_cen = np.random.randint(data.shape[0])
        centroids[0] = data[init_cen]
        #need to flatten() because it will remove extra dimension returned by pairwise distances used later in devide
        squared_distances = self.pairwse_euclidean_dis(data, centroids[0:1]).flatten()**2
        for i in range(1, k):
            # np.random.choice(range of numbers, size, probablity) pick the number from range of number whose assiciate
            # probablity is higher given in p list
            id_centroids = np.random.choice(data.shape[0], 1, p=squared_distances/sum(squared_distances))
            centroids[i] = data[id_centroids]
            squared_distances = np.min(self.pairwse_euclidean_dis(data, centroids[0:i+1])**2,axis=1)
        return centroids
    

    def kmeans(self, data, k, initial_centroids=None, maxiter=1000, verbose = False, seed=None):
        """ driver method to get centroids and cluster assignments from k-mean algorithm

            args :
                data (numpy array) : numpy array of observation points
                k (int) : number of clusters
                initial_centroids (numpy array) : initial centroids if explicitely provided (optional) if not used kmean++ for initialization
                maxiter (int) : maximum iteration to stop k mean algorithm reassignments infinite times (optional)
                verbose (bool) : print iteration and centroids info (optional)
                seed (int) : set seed for constant randomness fir initialization (optional)

            Returns :
                numpy array of centroids, numpy array of cluster assignments
        """

        centroids = initial_centroids[:] if initial_centroids is not None else self._k_meanpp_initialize(data, k, seed)
        prev_cluster_assignment = None 
        for itr in range(maxiter):        
            cluster_assignment = self._assign_clusters(data, centroids)
            centroids = self._update_centroids(data, k, cluster_assignment)
            # if no changes detected
            if prev_cluster_assignment is not None and (prev_cluster_assignment==cluster_assignment).all():
                break
            if verbose :
                print('Iteration '+ str(itr)+ ' centroids : ')
                print(centroids)
            prev_cluster_assignment = cluster_assignment[:]
        return centroids, cluster_assignment
    

    def get_initial_k(self, X, n, in_dict=True, seed=None):
        """ get n heterogeneity vs k value, used to select optimum value of k

            args :
                X (numpy array) : numpy array of observation points
                n (numpy array) : number of heterogeneity to be calculated
                in_dict (bool) : if true result will be in dictionary (optional)
                seed (int) : set seed for constant randomness fir initialization (optional)

            Returns :
                list of tuple or a dictionary having value of k and associate heterogeneity
        """

        if n > len(X) :
            raise Exception('n value is too large')
        heterogeneity = []
        k_list = []
        for k in range(1,n+1) :
            init_cent = self._k_meanpp_initialize(X, k, seed=seed)
            centroids, cluster_assignment =  self.kmeans(X, k, init_cent)
            heterogeneity.append(self.get_heterogeneity(X, k, centroids, cluster_assignment))
            k_list.append(k)
        return dict(zip(k_list, heterogeneity)) if in_dict else zip(k_list, heterogeneity)
    

