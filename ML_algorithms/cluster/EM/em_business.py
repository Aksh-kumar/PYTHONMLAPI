#!/usr/bin/env python
# coding: utf-8

import imageio
import os,re, base64, sys
import pandas as pd
import numpy as np
from .EM import EM
import pickle as pk
from ..KM import K_mean as KM
from ML_algorithms.supporting_module import pickle_module as pkl
sys.modules['K_mean'] = KM
sys.modules['pickle_module'] = pkl
"""
if module import need to do explicitely
#import importlib.util
#km_path = importlib.util.spec_from_file_location("K_mean", os.path.join("../KM/K_mean.py"))
#KM = importlib.util.module_from_spec(km_path)
#km_path.loader.exec_module(KM)
#pkl_path = importlib.util.spec_from_file_location("pickle_module", os.path.join("../../supporting_module/pickle_module.py"))
#pkl = importlib.util.module_from_spec(pkl_path)
#pkl_path.loader.exec_module(pkl)
"""
# constant used to store pickle file
EM_PICKLES_SAVE_SUB_DIR = r'\Pickle_EM' # saved sub directory of pickle file

""" EM business lgic class run EM algorithm from scratch using k-mean as initializer
 expected parameter training images path"""
class EMBusiness :
    def __init__(self, path=None) :
        self._dataset = None
        self._X = None
        self._k = None
        self._em_params = None
        self._dic_k_to_name = {}
        self.init_centroids = None
        self.init_cluster_assignment = None
        self.df = pd.DataFrame([],columns=['Image_Name', 'Path', 'Extension', 'R', 'G', 'B' ])
        self.image_EXT = ['.png', '.jpeg', '.jpg', '.jif', '.jpe']
        self._km = KM.K_Mean()
        self._em = EM()
        if path is not None :
            self._dataset = self.get_dataset_from_path(path)
            self.get_RGB_numpy_array()
    # End
    @property
    def em_parameters(self) :
        if self._em_params is None :
            raise Exception('Em parameter not found please run em_params method to get the parameters')
        else :
            return self._em_params.copy()
    # End
    @property
    def IMAGE_EXT_SUPPORTED(self) :
        return self.image_EXT
    # End
    @property
    def cluster_name(self) :
        return self._dic_k_to_name
    # End
    @cluster_name.setter
    def cluster_name(self, value) :
        if not (len(value.keys()) == self._k) :
            raise Exception('length must be the same as number of clusters')
        else :
            self._dic_k_to_name = value
    # End
    @property
    def dataset(self) :
        return self._dataset.copy() if self._dataset is not None else None
    # End
    @dataset.setter
    def dataset(self, value) :
        self._dataset = value
        self.get_RGB_numpy_array()
    # End
    @property
    def k(self) :
        return self._k
    # End
    @k.setter
    def k(self, value) :
        try :
            self._k = int(value)
        except :
            raise Exception('value must be Integer')
    # End
    def get_hard_assignment(self, resp=None, k=None, one_hot_encoded = False):
        if resp is not None and k is not None :
            assert resp.shape[1] == k
            res = list(map(lambda  x : int(np.where(x==np.max(x))[0]), resp))
            return np.array(res) if not one_hot_encoded else np.eye(k, dtype='int')[res]
        if resp is None and k is None :
            if self._em_params is None or self._k is None :
                raise Exception('Either of em_parameters or k (number of cluster) not found')
            res = list(map(lambda  x : int(np.where(x==np.max(x))[0]), self._em_params['responsibility']))
            return np.array(res) if not one_hot_encoded else np.eye(self._k, dtype='int')[res]
        raise Exception('some parameter not been found')
    # End   
    def _get_image_feature(self, path) :
        try :
            Im = imageio.imread(os.path.join(path), pilmode='RGB')
            temp = Im/255. # divide by 255 to get in fraction
            mn = temp.sum(axis=0).sum(axis=0)/(temp.shape[0]*temp.shape[1])
            return mn/np.linalg.norm(mn, ord=None)
        except :
            return None
    # End
    def _create_image_dataset(self, path, df = None) :
        dataset = self.df if df is None else df
        for files in os.listdir(os.path.join(path)) :
            f_path = os.path.join(path, files)
            if os.path.isfile(f_path) :
                temp = os.path.splitext(files)
                ext = temp[1]
                if str.lower(ext) in self.image_EXT :
                    f_name = temp[0]
                    mn = self._get_image_feature(f_path)
                    #     file_name, path,extension,red, green, blue
                    data = [f_name, f_path, ext, mn[0], mn[1], mn[2]]
                    data_dic = dict(zip(list(dataset.columns), data))
                    if df is not None :
                        dataset = dataset.append(data_dic, ignore_index=True)
            if os.path.isdir(f_path) :
                dataset = self._create_image_dataset(f_path , dataset)
        return dataset
    # End
    """ get RGB vector of image using imageio"""
    def get_RGB_numpy_array(self, dataset=None) :
        X = dataset.iloc[:, [3, 4, 5]].values if dataset is not None else self._dataset.iloc[:, [3, 4, 5]].values
        if dataset is None :
            self._X = X[:, :]    
        return X
    # End
    """ take path of directory in which training images are stored and
     return pandas dataframe of features """
    def get_dataset_from_path(self, path, num_rows = None, in_dic_list = False) :
        if not os.path.isdir(os.path.join(path)) :
            raise Exception('please provide directory path of training images')
        self._dataset = self._create_image_dataset(path)
        if in_dic_list :
            return self.dataset.T.to_dict().values() if num_rows is None else self.dataset.head(num_rows).T.to_dict().values()
        else :
            return self.dataset if num_rows is None else self.dataset.head(num_rows)
    # End
    """ parameter n is number of clusters with there associate heterogeneity and data is numpy array"""
    def get_first_n_heterogeneity(self, n, data=None) :
        data = data if data is not None else self._dataset
        if data is None :
            raise Exception('No data found')
        X = self._X if data is None else data
        return self._km.get_initial_k(X, n)
    # End
    """ since cluster assignment produces hard assignment through k mean this fur computational
     stability we added small amount of reponsibility to not assigned cluster because for EM
     we have to take inverse of covariance metrix if it become a singular metrix it will not 
     get inverted"""
    def _get_stable_responsibilities(self, resp, k=None) :
        k = k if k is not None else self._k
        # Inner function
        def get_soft_assignment_list(ls, n) :
            temp = np.random.uniform(low=0.1, high=0.16,size =(n,))
            index_1 = int(np.where(ls==1)[0])
            sum_ele = temp.sum()
            return np.insert(temp, index_1, 1-sum_ele)
        return np.array(list(map(lambda y : get_soft_assignment_list(y, k-1), resp)))
    # End
    """ expected parameter k- number of cluster and data is numpy array of feature is optional if
     not given then the dataset created through this object  RBG value is getting fetched"""
    def get_initial_centroids_and_cluster_assignment(self, k=None, data=None, seed = None) :
        data_temp = data if data is not None else self._X
        k = k if k is not None else self._k
        if data_temp is None :
            raise Exception('No data found please use get_RGB_numpy_array(data(pandas dataframe)) to get RGB numpy array')
        if k is None :
            raise Exception('No value of k found please set value of k i.e object.k = value')
        centroids, cluster_assignments = self._km.kmeans(data_temp, k, seed = seed)
        if data is None and self._X is not None :
            self.init_centroids = centroids
            self.init_cluster_assignment = np.eye(k)[cluster_assignments]
        return centroids, np.eye(k)[cluster_assignments]
    # End
    """ parameter data numpy array, responsibility numpy list of responsibility, maxiteer maximum iteration of
     EM algorithm threshold - stopping threshold value if value not change further this"""
    def get_em_params(self, data=None, resp=None, maxiter = 1000, thresh=1e-4, seed = None) :
        if data is None and self._X is None :
            raise Exception('No data found')
        elif data is None and self._X is not None :
            # set initial centroids and assignments and hard assignments to clusters by k-means
            self.get_initial_centroids_and_cluster_assignment(seed=seed)
            resp = self._get_stable_responsibilities(self.init_cluster_assignment)
            self._em_params = self._em.em(self._X, resp, max_iter=maxiter, threshold=thresh)
            return self.em_parameters
        elif data is not None and resp is not None and self._X is None :
            return self._em.em(data, resp, max_iter=maxiter, threshold=thresh)
        else :
            return None
    # End
    """ predict the soft assignment of given data if means, covariance and weight 
     is not given then internal saved em output evaluated from dataset is used"""
    def predict_soft_assignments(self, data, means=None, covariance=None, weight=None) :
        if means is None and covariance is None and weight is None :
            if self._X is None :
                raise Exception('can\'t find the data please provide training data')
            if self._k is None :
                raise Exception('can\'t find the value of k')
            if self._em_params is None :
                raise Exception('EM parameter is not being set')
            if not (data.shape[1] == self._X.shape[1]) :
                raise Exception('dimention of data mismatch')
            return self._em.get_responsibilities(data, self._em_params['weights'],
                                                 self._em_params['means'],
                                                 self._em_params['covariances'])
        if means is not None and covariance is not None and weight is not None : 
            n_cluster = means.shape[0]
            dimension = means.shape[1]
            if weight.ndim == 1 and weight.shape[0] == n_cluster and \
                covariance.ndim == 3 and covariance.shape[0] == n_cluster and \
                covariance.shape[1] == dimension and covariance.shape[2] == dimension :
                return self._em.get_responsibilities(data, weight, means, covariance)
            else :
                raise Exception('Dimension of paramenter do not match')
        else :
            raise Exception('some parameter is missing')
    # End
    def get_first_n_data_responsibility(self, n, emobj = None, to_dict=False) :
        emobj = emobj if emobj is not None else self
        df = emobj.dataset
        len_dataframe = len(df.index)
        if n > len_dataframe :
            raise Exception('required number of data is larger than available size of dataset')
        else :
            result = {}
            resp = pd.DataFrame(emobj.em_parameters['responsibility'], columns= list(range(emobj.k)))
            hard_assign = pd.DataFrame(emobj.get_hard_assignment(), columns = ['Assign_Cluster'])
            df = pd.concat([df, resp, hard_assign], axis=1)
            for i in range(emobj.k) :
                temp = df[df['Assign_Cluster'] == i].head(n)
                base64_list = []
                for row in temp.iterrows() :
                    encd_64 = pkl.encode_base64(row[1]['Path'])
                    base64_list.append(encd_64)
                temp['Image_base64'] = base64_list
                result[i] = temp if not to_dict else temp.T.to_dict().values()
            return result
    # End
# End class

def get_em_object(k, TRAINING_PATH_DIR, SEED = 0) :
    # replace pickle file name non alphabatical character to _
    f_name = re.sub('[^0-9a-zA-Z]+', '_', TRAINING_PATH_DIR) +'_' + str(k) + '.pickle'
    pkl_obj = pkl.read_pickled_object(EM_PICKLES_SAVE_SUB_DIR, f_name)
    if pkl_obj is None :
        em_obj = EMBusiness(TRAINING_PATH_DIR)
        em_obj.k = k
        em_obj.get_em_params(seed=SEED)
        pkl_obj = pkl.GetPickledObject.get_EM_pickled(TRAINING_PATH_DIR, em_obj, k)
        pkl.write_pickled_object(pkl_obj, EM_PICKLES_SAVE_SUB_DIR, f_name)
    return pkl_obj
# End
