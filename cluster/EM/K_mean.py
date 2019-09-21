import numpy as np
# Maximum limit of number of heterogeneity required
MAX_VAL = 1000
def check_max_k(func) :
    def inner(*args) : 
        if args[2] > MAX_VAL :
           raise Exception("max value greater then {0} is not allowed".format(MAX_VAL))
        else :
            return func(*args)
    return inner
# End
class K_Mean :
    def __init__(self) :
        pass
    @staticmethod
    def pairwse_euclidean_dis(data, centroids) :
        assert data.shape[1] == centroids.shape[1]
        dis_list = list(map(lambda centroid : (np.sum((data - centroid)**2, axis=1)**0.5)[:, np.newaxis], centroids))
        return np.concatenate(dis_list, axis=1)
    # End
    def _assign_clusters(self, data, centroids):
        distances_from_centroids = self.pairwse_euclidean_dis(data, centroids)
        cluster_assignment = np.argmin(distances_from_centroids,axis=1)
        return cluster_assignment
    # End
    def _update_centroids(self, data, k, cluster_assignment):
        new_centroids = []
        for i in range(k):
            cluster_points = data[cluster_assignment==i]
            centroid = cluster_points.mean(axis=0)
            centroid = centroid.ravel()
            new_centroids.append(centroid)
        return np.array(new_centroids)
    # End
    def get_heterogeneity(self, data, k, centroids, cluster_assignment):
        heterogeneity = 0.0
        for i in range(k):
            cluster_points = data[cluster_assignment==i, :]
            if cluster_points.shape[0] > 0: # check if i-th cluster is non-empty
                distances = self.pairwse_euclidean_dis(cluster_points, np.array([centroids[i]]))
                squared_distances = distances**2
                heterogeneity += np.sum(squared_distances)
        return heterogeneity
    # End
    def _k_meanpp_initialize(self, data, k, seed=None):
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
    # End
    def kmeans(self, data, k, initial_centroids=None, maxiter=1000, verbose = False, seed=None):
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
    # End
    @check_max_k
    def get_initial_k(self, X, max_k=15, in_dict=True) :
        heterogeneity = []
        k_list = []
        for k in range(1,max_k+1) :
            init_cent = self._k_meanpp_initialize(X, k, seed=0)
            centroids, cluster_assignment =  self.kmeans(X, k, init_cent)
            heterogeneity.append(self.get_heterogeneity(X, k, centroids, cluster_assignment))
            k_list.append(k)
        return dict(zip(k_list, heterogeneity)) if in_dict else zip(k_list, heterogeneity)
    # End
# End class
