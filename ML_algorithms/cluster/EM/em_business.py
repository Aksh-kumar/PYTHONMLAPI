#!/usr/bin/env python
# coding: utf-8

import imageio, json
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
EM_CLUSTER_NAME_SAVE_SUB_DIR = r'Cluster_Name_EM'
""" EM business lgic class run EM algorithm from scratch using k-mean as initializer
 expected parameter training images path"""
class EMBusiness :
    def __init__(self, kmObj, emObj, path=None, cluster_name_json_file_path=None) :
        self._dataset = None
        self._X = None
        self._k = None
        self._em_params = None
        self.init_centroids = None
        self.init_cluster_assignment = None
        self._col_list = ['ImageName', 'Path', 'Extension', 'R', 'G', 'B' ]
        self.base64_col_name = 'ImageBase64'
        self._assign_cluster_col_name = 'AssignCluster'
        self.df = pd.DataFrame([],columns=self._col_list)
        self.image_EXT = ['.png', '.jpeg', '.jpg', '.jif', '.jpe']
        self._km = kmObj
        self._em = emObj
        self.cluster_name_path = cluster_name_json_file_path
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
        if self.cluster_name_path is not None and os.path.exists(self.cluster_name_path) :
            with open(os.path.join(self.cluster_name_path), 'r') as f :
                return json.load(f)
        else :
            return json.loads(json.dumps({x:str(x) for x in range(self.k)}))
    # End
    def set_cluster_name(self, value) :
        if not (len(value.keys()) == self._k) :
            raise Exception('length must be the same as number of clusters')
        else :
            try :
                with open(os.path.join(self.cluster_name_path), 'w') as f :
                    json.dump(value, f)
                return json.loads(json.dumps({'res': True}))
            except :
                return json.loads(json.dumps({'res': False}))
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
    def _get_base_dataset(self, filename, f_path, ext, R, G, B) :
        dic = dict(zip(self._col_list, [[filename], [f_path], [ext], [R], [G], [B]]))
        return pd.DataFrame(dic)
    # End
    def _shuffle_column(self, df) :
        init_col = df.columns.tolist()
        final_col = []
        res_col = []
        for i in init_col :
            try :
                res_col.append(int(i))
            except ValueError :
                final_col.append(i)
        final_col.extend(res_col)
        df = df[final_col]
        return df
    # End
    def get_hard_assignment(self, resp=None, k=None, one_hot_encoded = False, to_pandas_DF = False):
        if resp is not None and k is not None :
            assert resp.shape[1] == k
            res = list(map(lambda  x : int(np.where(x==np.max(x))[0]), resp))
            result = np.array(res) if not one_hot_encoded else np.eye(k, dtype='int')[res]
            return pd.DataFrame(result, columns = [self._assign_cluster_col_name]) if to_pandas_DF else result
        if resp is None and k is None :
            if self._em_params is None or self._k is None :
                raise Exception('Either of em_parameters or k (number of cluster) not found')
            res = list(map(lambda  x : int(np.where(x==np.max(x))[0]), self._em_params['responsibility']))
            result = np.array(res) if not one_hot_encoded else np.eye(self._k, dtype='int')[res]
            return pd.DataFrame(result, columns = [self._assign_cluster_col_name]) if to_pandas_DF else result
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
    def get_dataset_from_path(self, path, num_rows = None) :
        if not os.path.isdir(os.path.join(path)) :
            raise Exception('please provide directory path of training images')
        self._dataset = self._create_image_dataset(path)
        return self.dataset if num_rows is None else self.dataset.head(num_rows)
    # End
    """ parameter n is number of clusters with there associate heterogeneity and data is numpy array"""
    def get_first_n_heterogeneity(self, n, data=None, seed = None) :
        X = self._X if data is None else data
        if X is None :
            raise Exception('No data found')
        return self._km.get_initial_k(X, n=n, seed=seed)
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
    """ parameter data numpy array, maxiteer maximum iteration of
     EM algorithm threshold - stopping threshold value if value not change further this"""
    def get_em_params(self, data=None, maxiter = 1000, thresh=1e-4, seed = None) :
        if data is None and self._X is None :
            raise Exception('No data found')
        elif data is None and self._X is not None :
            # set initial centroids and assignments and hard assignments to clusters by k-means
            self.get_initial_centroids_and_cluster_assignment(seed=seed)
            self._em_params = self._em.em(self._X, self.init_centroids , max_iter=maxiter, threshold=thresh)
            return self.em_parameters
        elif data is not None and self._X is None :
            centroids, cluster_assignments = self.get_initial_centroids_and_cluster_assignment(data = data,seed=seed)
            return self._em.em(data, centroids, max_iter=maxiter, threshold=thresh)
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
    def get_first_n_data_responsibility(self, n, to_json=False) :
        emobj = self
        df = emobj.dataset
        len_dataframe = len(df.index)
        if n > len_dataframe :
            raise Exception('required number of data is larger than available size of dataset')
        else :
            result = {}
            resp = pd.DataFrame(emobj.em_parameters['responsibility'], columns= list(range(emobj.k)))
            hard_assign = emobj.get_hard_assignment(to_pandas_DF=True)
            df = pd.concat([df, resp, hard_assign], axis=1)
            for i in range(emobj.k) :
                temp = df[df[self._assign_cluster_col_name] == i]#.head(n)
                temp = temp.sort_values(by=i, ascending=False)
                temp = temp.head(n)
                base64_list = []
                for row in temp.iterrows() :
                    encd_64 = pkl.encode_base64(row[1]['Path'])
                    base64_list.append(encd_64)
                temp[self.base64_col_name] = base64_list
                temp = self._shuffle_column(temp)
                temp = temp.drop(['Path'], axis=1)
                result[i] = temp.to_json(orient='records') if to_json else temp
            return result
    # End
    def predict_data(self, filename, filetype, filepath, val_base64, means=None, covariance=None, weight=None) :
        ext = '.' + filetype.split('/')[1]
        temp = self._get_image_feature(filepath)
        R, G, B = temp[0], temp[1], temp[2]
        resp = self.predict_soft_assignments(np.array([temp]), means=None, covariance=None, weight=None)
        hard_assign = self.get_hard_assignment(resp, self._k, to_pandas_DF=True)
        # hard_assign = hard_assign.loc[:, (hard_assign != 0).any(axis=0)]
        resp = pd.DataFrame(resp, columns= list(range(self.k)))
        df = self._get_base_dataset(filename, filepath, ext, R, G, B)
        df[self.base64_col_name] = val_base64
        df = pd.concat([df, resp, hard_assign], axis=1)
        df = self._shuffle_column(df)
        df = df.drop(['Path'], axis=1)
        return df
# End class

# replace pickle file name non alphabatical character to _
# get file Name
get_file_name = lambda k, pth, ext : re.sub('[^0-9a-zA-Z]+', '_', pth) +'_' + str(k) + ext

# write E pickle
# parameter k-number of cluster, training path directory path, seed seed to initialize cluster parameter
# and f_name file name to save as pickle
def write_em_pickle(k, TRAINING_PATH_DIR, seed, f_name=None) :
    f_name = f_name if f_name is not None else get_file_name(k, TRAINING_PATH_DIR, '.pickle')
    kmObj = KM.K_Mean()
    emObj = EM()
    f_name_json = f_name.split('.')[0] + '.json' if f_name is not None else get_file_name(k, TRAINING_PATH_DIR, '.json')
    jsonpath = pkl.get_json_file_path(f_name_json, EM_CLUSTER_NAME_SAVE_SUB_DIR)
    em_obj = EMBusiness(kmObj, emObj, TRAINING_PATH_DIR, jsonpath)
    em_obj.k = k
    em_obj.get_em_params(seed=seed)
    pkl_obj = pkl.GetPickledObject.get_EM_pickled(TRAINING_PATH_DIR, em_obj, k)
    pkl.write_pickled_object(pkl_obj, EM_PICKLES_SAVE_SUB_DIR, f_name)
    return pkl_obj
# End
# get EM object
def get_em_object(k, TRAINING_PATH_DIR, seed = None, saved_model_name=None) :
    f_name = saved_model_name if saved_model_name is not None else get_file_name(k, TRAINING_PATH_DIR, '.pickle')
    pkl_obj = pkl.read_pickled_object(EM_PICKLES_SAVE_SUB_DIR, f_name)
    if pkl_obj is None :
        pkl_obj = write_em_pickle(k, TRAINING_PATH_DIR, seed, f_name)
    return pkl_obj
# End
