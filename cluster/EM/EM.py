import numpy as np
from scipy.stats import multivariate_normal
from functools import reduce

class EM :
    # get sum of all responsibilty for a particular cluster
    # resp = nxk metrix i.e resp[number_of_label, nuber_of_cluster]
    get_soft_counts = lambda resp : np.sum(resp, axis=0)
    def __init__(self) :
        pass
    # compute ln(exp^e1+expe2....)
    def _ln_sum_exp(self, Z):
        return np.log(np.sum(np.exp(Z)))
    # End
    def get_log_likelihood(self, data, weights, means, covs):
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
    # End
    # E step - get the responsibility if cluster parameter is given
    def get_responsibilities(self, data, weights, means, covariances):
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
    # End
    #M-step update the parameter if given cluster responsibilty is given
    # get list of weights which represents how much each cluster represented over all data points
    # parameter count list of sum of soft counts for all clusters
    def _get_weights(self, counts):
        n_clusters = len(counts)
        sum_count = np.sum(counts)
        weights = np.array(list(map(lambda k : counts[k]/sum_count, range(n_clusters))))
        return weights
    # End
    def _get_means(self, data, resp, counts):
        n_clusters = len(counts)
        n_data = len(data)
        means = np.zeros((n_clusters, len(data[0])))
        for k in range(n_clusters):
            weighted_sum = reduce(lambda x,i : x + resp[i,k]*data[i], range(n_data), 0.0)
            means[k] = weighted_sum/counts[k]
        return means
    # End
    # update covariances metrix from responsibity metrix, means and counts
    def _get_covariances(self, data, resp, counts, means):
        n_clusters = len(counts)
        dimension = len(data[0])
        n_data = len(data)
        covariances = np.zeros((n_clusters, dimension, dimension))
        for k in range(n_clusters):
            weighted_sum = reduce(lambda x, i :x + resp[i,k] * np.outer((data[i]-means[k]), (data[i]-means[k]).T),
                                range(n_data), np.zeros((dimension, dimension)))
            covariances[k] = weighted_sum/counts[k]
        return covariances
    # End
    # EM algorithm
    def em_from_parameter(self, data, init_means, init_covariances, init_weights, maxiter=1000, thresh=1e-4, verbose=False):
        # Make copies of initial parameters
        means = init_means[:]
        covariances = init_covariances[:]
        weights = init_weights[:]
        # Infer dimensions of dataset and the number of clusters
        n_data = len(data)
        # dimension = len(data[0])
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
            counts = EM.get_soft_counts(resp)
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
    # End
    def em_initializer_from_responsibility(self, data, resp) :
        counts = EM.get_soft_counts(resp)
        weights = self._get_weights(counts)
        means = self._get_means(data, resp, counts)
        covariances = self._get_covariances(data, resp, counts, means)
        return means, covariances, weights
    # End
    def em(self, data, resp, max_iter = 1000, threshold = 1e-4) :
        means, covariances, weights = self.em_initializer_from_responsibility(data, resp)
        return self.em_from_parameter(data, means, covariances, weights, maxiter=max_iter, thresh=threshold)
# End class

