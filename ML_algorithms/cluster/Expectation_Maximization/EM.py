#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.stats import multivariate_normal
from functools import reduce

class EM:
    """ Low level Implementation of Gaussian Mixture
        model for likelihood estimation
    """
    
    def __init__(self):

        pass # nothing to initialize


    def get_soft_counts(self, resp):
        """ get sum of all responsibilty for a particular cluster
            resp = nxk metrix i.e resp[number_of_label, number_of_cluster]

            args :
                resp(numpy array) : responsibility  numpy array size (n_sample, n_clusters)
            
            Returns :
                numpy array of sum of all soft counts size (1, n_clusters)
        """

        return np.sum(resp, axis=0)
    
    
    def _ln_sum_exp(self, Z):
        """ compute ln(exp^e1+expe2....)
        
            args :
                Z (numpy array) : numpy array of size (1, n_clusters)
            
            Returns :
                float value of log of sum of exponential
        """

        return np.log(np.sum(np.exp(Z)))
    

    def get_log_likelihood(self, data, weights, means, covs):
        """ calculate multivariate_normal distribution value
        from data weight of cluster mean and covariance from
        multivarialte normal distribution formula and return sum
        of log of exponential

            args :
                data (numpy array) : array of observation points
                weights (numpy array) : numpy array of weight of each clusters ofsize (1, n_clusters)
                means (numpy array) : numpy array of means of each clusters of size (n_cluster, dimension)
                covs (numpy array) : numpy array of covariance metrix of size (n_clusters, dimension, dimension)
            
            Returns :
                float value of log likelihood
        """

        n_clusters = len(means)
        dimension = len(data[0]) 
        sum_ln_exp = 0
        for d in data:
            Z = np.zeros(n_clusters)
            for k in range(n_clusters):
                # compute exponential term in multivariate_normal
                delta = np.array(d) - means[k]
                inv = np.linalg.inv(covs[k])
                exponent_term = np.dot(delta.T, np.dot(inv, delta))
                # Compute loglikelihood contribution for this data point and this cluster
                Z[k] += np.log(weights[k])
                det = np.linalg.det(covs[k])
                Z[k] -= 1/2. * (dimension * np.log(2*np.pi) + np.log(det) + exponent_term)
            # Increment loglikelihood contribution of this data point across all clusters
            sum_ln_exp += self._ln_sum_exp(Z)
        return sum_ln_exp
    
    # E step - get the responsibility if cluster parameter is given

    def get_responsibilities(self, data, weights, means, covariances):
        """ take cluster parameter as input and calculate
            responsibility by using multivariate normal distribution
            
            args :
                data (numpy array) : array of observation points
                weights (numpy array) : numpy array of weight of each clusters ofsize (1, n_clusters)
                means (numpy array) : numpy array of means of each clusters of size (n_cluster, dimension)
                covs (numpy array) : numpy array of covariance metrix of size (n_clusters, dimension, dimension)
            
            Returns :
                numpy array of responsibility of each data points
        """

        n_data = len(data)
        n_clusters = len(means)
        resp = np.zeros((n_data, n_clusters))
        for i in range(n_data):
            for k in range(n_clusters):
                resp[i, k] = weights[k]* multivariate_normal.pdf(data[i],means[k],covariances[k],allow_singular=True)
        # Add up responsibilities over each data point and normalize
        row_sums = resp.sum(axis=1)[:, np.newaxis]
        resp = resp / row_sums
        return resp
    
    
    # M-step update the parameter if given cluster responsibilty is given
    
    def _get_weights(self, counts):
        """ get list of weights which represents how much each
            cluster represented over all data points parameter

            args :
                counts (numpy array) : count list of sum of soft counts for all clusters of size (n_cluster)

            Returns :
                weights (numpy array) : numpy array of weight of each clusters ofsize (1, n_clusters)
        """

        n_clusters = len(counts)
        sum_count = np.sum(counts)
        weights = np.array(list(map(lambda k : counts[k]/sum_count, range(n_clusters))))
        return weights
    

    def _get_means(self, data, resp, counts):
        """ take responsibility and sum of soft counts for all clusters
            and calculate the mean corrosponding to each cluster

            args :
                data (numpy array) : array of observation points
                resp(numpy array) : responsibility  numpy array size (n_sample, n_clusters)
                counts (numpy array) : count list of sum of soft counts for all clusters of size (n_cluster)
            
            Returns :
                numpy array of means of each clusters of size (n_cluster, dimension)
        """

        n_clusters = len(counts)
        n_data = len(data)
        means = np.zeros((n_clusters, len(data[0])))
        for k in range(n_clusters):
            weighted_sum = reduce(lambda x,i : x + resp[i,k]*data[i], range(n_data), 0.0)
            means[k] = weighted_sum/counts[k]
        return means
    

    def _get_covariances(self, data, resp, counts, means):
        """ calculate covariances metrix from
            responsibity metrix, means and counts

            args :
                data (numpy array) : array of observation points
                resp(numpy array) : responsibility  numpy array size (n_sample, n_clusters)
                counts (numpy array) : count list of sum of soft counts for all clusters of size (n_cluster)
                means (numpy array) : numpy array of means of each clusters of size (n_cluster, dimension)
        
            Returns :
                numpy array of covariance metrix of size (n_clusters, dimension, dimension)
        """

        n_clusters = len(counts)
        dimension = len(data[0]) # to get dimention of data
        n_data = len(data)
        covariances = np.zeros((n_clusters, dimension, dimension))
        for k in range(n_clusters):
            # outer product of [a0, a1, a2] * [b0, b1, b2] = [[a0*b0, a0*b1, a0*b2], [a1*b0, a1*b1, a1*b2], [a2*b0, a2*b1, a2*b2]]
            # outer_product = ((data_i - mean_k) * (data_i - mean_k).T)
            # compute weighted_sum = (SUM)i=0 to N(no of sample) responsibility(ik) * outer_product
            weighted_sum = reduce(lambda x, i :x + resp[i,k] * np.outer((data[i]-means[k]), (data[i]-means[k]).T),
                                range(n_data), np.zeros((dimension, dimension)))
            covariances[k] = weighted_sum/counts[k] # normalize by total sum of counts
        return covariances


    def em_from_parameter(self, data, init_means, init_covariances, init_weights, maxiter=1000, thresh=1e-4, verbose=False):
        """
            EM algorithm implementation which take initial mean, covariance and weights as initial parameter
            and run till max_iter or tilll convergence if difference between previous and current parameter
            estimation less then thresh value

            args :
                data (numpy array) : array of observation points
                init_weights (numpy array) : numpy array of initial weight of each clusters ofsize (1, n_clusters)
                init_means (numpy array) : numpy array of initial means of each clusters of size (n_cluster, dimension)
                init_covariances (numpy array) : numpy array of initial covariance metrix of size (n_clusters, dimension, dimension)
                maxiter (int) : maximum iteration to rum EM (optional)
                threshold (float) : maximum threshold to stop EM (optional)
                verbose (bool) : if true then print intermediate value (optional)

            Returns :
                object contains  {'weights': weights of clusters, 'means': means of clusters, 'covariances': covariances of clusters,
                'loglikelihood': log likelihood list, 'responsibility': responsibility of data points ,
                'Iterations': maximum iteration taken for convergence}
        """

        # Make copies of initial parameters
        means = init_means[:]
        covariances = init_covariances[:]
        weights = init_weights[:]
        # Infer length of dataset
        n_data = len(data)
        # Infer number of cluster
        n_clusters = len(means)
        # Initialize some useful variables
        resp = np.zeros((n_data, n_clusters), dtype=np.float64)
        ll = self.get_log_likelihood(data, weights, means, covariances)
        ll_list = [ll]
        for it in range(maxiter):
            if verbose and it % 5 == 0:
                print("Iteration %s" % it)
            # E-step: calculate responsibilities
            resp = self.get_responsibilities(data, weights, means, covariances)
            # M-step calculate cluster parameter
            counts = self.get_soft_counts(resp)
            weights = self._get_weights(counts)
            means = self._get_means(data, resp, counts)
            covariances = self._get_covariances(data, resp, counts, means)
            ll_new = self.get_log_likelihood(data, weights, means, covariances)
            ll_list.append(ll_new)
            if abs(ll_new - ll) < thresh :
                break
            ll = ll_new
        out = {'weights': weights, 'means': means, 'covariances': covariances,
         'loglikelihood': ll_list, 'responsibility': resp, 'Iterations': it}
        return out
    

    def _em_initializer_from_responsibility(self, data, init_centroids):
        """ get initial parameter to start EM algorithms

            args :
                data (numpy array) : array of observation points
                init_centroids (numpy array) : initial centroids of cluster of size (n_cluster, dimension)

            Returns :
                numpy array of initial mean, covariance and weights
        """

        means = init_centroids # mean co-ordinate of all clusters
        n_cluster = len(init_centroids) # get number f clusters
        cov = np.diag(np.var(data, axis=0)) # get covariance of data initialize by diagonal metrix of covariance of data
        covariances = np.array([cov]*n_cluster) # initiate covariance metrix with diagonal element is covariance of data
        weights = np.array([1/n_cluster]*n_cluster) # initialize equal weight to all clusters
        return means, covariances, weights
    

    def em(self, data, init_centroids, max_iter = 1000, threshold = 1e-4):
        """
            driver function to calculate cluster responsibility and parameters
            internally call _em_initializer_from_responsibility and then em_from_parameter
            
            args :
                data (numpy array) : array of observation points
                init_centroids (numpy array) : initial centroids of cluster of size (n_cluster, dimension)
                maxiter (int) : maximum iteration to rum EM (optional)
                threshold (float) : maximum threshold to stop EM (optional)
            
            Returns :
                object contains  {'weights': weights of clusters, 'means': means of clusters, 'covariances': covariances of clusters,
                'loglikelihood': log likelihood list, 'responsibility': responsibility of data points ,
                'Iterations': maximum iteration taken for convergence}
        """

        means, covariances, weights = self._em_initializer_from_responsibility(data, init_centroids)
        return self.em_from_parameter(data, means, covariances, weights, maxiter=max_iter, thresh=threshold)



