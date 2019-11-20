#!/usr/bin/env python
# coding: utf-8

import json
import os
import re
import sys
import imageio
import pandas as pd
import numpy as np
from ML_algorithms.cluster.Expectation_Maximization.EM import EM
from ML_algorithms.cluster.K_mean.KM import KM
from ML_algorithms.supporting_module import pickle_module as pkl
import ML_algorithms.constants as CONST
sys.modules['pickle_module'] = pkl


class EMBusiness:
    """ EM business logic class run EM algorithm on Image Datasets
    """

    def __init__(self, kmobj, emobj, train_dir_path=None, cluster_name_json_file_path=None) :
        """ Initialize parameter
            
            args: 
                kmobj (Object): KMean Object
                emobj (Object): EM Object
                path (str): training folder path (optional)
                cluster_name_json_file_path (str): cluster number to name mapping json file path (optional)

        """
        self._dataset = None # pandas dataFrame store dataset of training data
        self._X = None # pandas dataframe stores image R,G,B features
        self._k = None # store number of cluster
        self._em_params = None # store EM algorithm trained parameter
        self.init_centroids = None # store Initail centroids obtained from k-mean
        self.init_cluster_assignment = None # contain sparse metrix of initial assigned cluster
        self._col_list = CONST.DATASET_COL # list of columns in _dataset
        self.base64_col_name = CONST.COL_BASE64_NAME # base 64 column name in _dataset
        self._assign_cluster_col_name = CONST.COL_ASSIGN_CLUSTER # assigned cluster number in _dataset
        self.df = pd.DataFrame([],columns=self._col_list) # creating empty dataframe
        self.image_EXT = CONST.SUPPORTED_IMAGES_EXTENSION # contain all the image extension supported for processing
        self._km = kmobj # referring dependency injector of KMean Object
        self._em = emobj # referring dependency injector for EM Object
        self.cluster_name_path = cluster_name_json_file_path # store file name with path for save cluster number to name mapping json
        if train_dir_path is not None : # if training path is available then create data Frame and fetch R, G, B vector
            self._dataset = self.get_dataset_from_path(train_dir_path)
            self.get_RGB_numpy_array()


    @property
    def em_parameters(self):
        """get property of class to get EM
            trained parameters DataFrame
            
            Returns :
                pandas DataFrame of EM parameters
        """

        if self._em_params is None :
            raise Exception('Em parameter not found please run em_params method to get the parameters')
        else :
            return self._em_params.copy()
    

    @property
    def IMAGE_EXT_SUPPORTED(self):
        """get supported Image Extension list
            
            Returns:
                list: Image extension in string
        """

        return self.image_EXT


    @property
    def cluster_name(self):
        """return cluster name json from json file if not
            exist then return same number mapped json
            
            Returns:
                Dictionary: cluster number to name mapping
        """

        if self.cluster_name_path is not None and os.path.exists(self.cluster_name_path) :
            with open(os.path.join(self.cluster_name_path), 'r') as f :
                return json.load(f)
        else :
            return json.loads(json.dumps({x:str(x) for x in range(self.k)})) # return same key mapped as cluster name
    

    def set_cluster_name(self, value):
        """ save cluster number to name mapping
            json provided from outside
            
            args:
                value(Dict): cluster number to name mapping
            
            Returns:
                json: {'res': bool(true if successfully saved else false)}
        """

        if not (len(value.keys()) == self._k) :
            raise Exception('length must be the same as number of clusters')
        else :
            try :
                with open(os.path.join(self.cluster_name_path), 'w') as f :
                    json.dump(value, f)
                return json.loads(json.dumps({'res': True}))
            except :
                return json.loads(json.dumps({'res': False}))
    

    @property
    def dataset(self):
        """ get dataset property to get train
            dataset copy if available else None
        
            Returns:
                Pandas DataFrame: pandas dataframe of training dataset
        """

        return self._dataset.copy() if self._dataset is not None else None
    

    @dataset.setter
    def dataset(self, value):
        """set dataset property to set train
            dataset if wanted to rovide explicitely
            but need to be in proper order as Column
            order mention in CONST
            
            args:
                value (Pandas DataFrame) : training Dataset
        """

        self._dataset = value
        self.get_RGB_numpy_array()
    

    @property
    def k(self):
        """ get property of number of cluster
            
            Returns:
                k (int) : number of cluster
        """

        return self._k    


    @k.setter
    def k(self, value):
        """ set property of number of cluster
            
            args:
                value (int) : number of cluster
        """

        try :
            self._k = int(value)
        except :
            raise Exception('value must be Integer')
    

    def _get_base_dataset(self, filename, f_path, ext, R, G, B):
        """ get pandas dataframe of provided data in parameters
            
            args :
                filename (str) : filename
                f_path (str) : file path
                ext (str) : file extension
                R (float) : Average Red intensity
                G (float) : Average Green intensity
                B (float) : Average Blue intensity
            
            Returns :
                Pandas DataFrame of observation data
        """

        dic = dict(zip(self._col_list, [[filename], [f_path], [ext], [R], [G], [B]]))
        return pd.DataFrame(dic)
    

    def _shuffle_column(self, df) :
        """ return the dataFrame with all base column and cluster responsibility with order
            
            args:
                df (Pandas DataFrame) : pandas Dataframe with random column order
            
            Returns :
                Pandas DataFrame : pandas Dataframe with ordered column
        """
        
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
    

    def get_hard_assignment(self, resp=None, k=None, one_hot_encoded = False, to_pandas_DF = False):
        """ get numpy array of soft assignment responsibility and find
            maximum responsibility of the cluster assignments number
            
            args:
                resp (numpy array) : array of responsibilities (optional)
                k (int) : number of cluster (optional)
                one_hot_encoded (bool) : if true then return one hot encoded value (optional)
                to_pandas_DF (bool) : if true then return pandas dataframe (optional)
                if argument is not provided then internal training dataset is used
            
            Returns: 
                Numpy 1D array or one hot encoded araay or Pandas DataFrame of resultant cluster assignment
         """
        
        if resp is not None and k is not None :
            assert resp.shape[1] == k # stop if responsibility column is not equal to number of cluster
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
   

    def _get_image_feature(self, path):
        """ get image R, G, B feature vector of Image from Image path
            
            args:
                path (str) : path of image
            
            Returns:
                Numpy array of [Red, Blue, Green] with l2 norm
        """
        try :
            Im = imageio.imread(os.path.join(path), pilmode='RGB')
            temp = Im/255. # divide by 255 to get in fraction
            mn = temp.sum(axis=0).sum(axis=0)/(temp.shape[0]*temp.shape[1])
            return mn/np.linalg.norm(mn, ord=None)
        except :
            return None
    

    def _create_image_dataset(self, path, df = None):
        """ create Image Pandas Dataframe from Image Path
            Folder return dataset of R, G, B vector of image
            
            args:
                path (str) : path of directory contain training images
                df (Pandas DataFrame) : Pandas DataFrame provided explicitely with propert column order (optional)
            
            Returns :
                Pandas DataFrame with filled the column details
        """

        dataset = self.df if df is None else df
        for files in os.listdir(os.path.join(path)) :
            f_path = os.path.join(path, files)
            if os.path.isfile(f_path) :
                temp = os.path.splitext(files)
                ext = temp[1] # get Image Extension
                if str.lower(ext) in self.image_EXT :
                    f_name = temp[0]
                    mn = self._get_image_feature(f_path)
                    #   file_name, path,extension,red, green, blue
                    data = [f_name, f_path, ext, mn[0], mn[1], mn[2]]
                    data_dic = dict(zip(list(dataset.columns), data))
                    if df is not None :
                        dataset = dataset.append(data_dic, ignore_index=True)
            if os.path.isdir(f_path) :
                dataset = self._create_image_dataset(f_path , dataset)
        return dataset
    

    def get_RGB_numpy_array(self, dataset=None):
        """ get RGB vector of image using imageio
            
            args:
                dataset (Pandas DataFrame) : return R, G, B  column from dataset
                if dataset is not provided returned dataset stored interenally and used later
            
            Returns :
                Pandas DataFrame contain only R, G, B values
        """

        X = dataset.iloc[:, [3, 4, 5]].values if dataset is not None else self._dataset.iloc[:, [3, 4, 5]].values
        if dataset is None :
            self._X = X[:, :]    
        return X


    def get_dataset_from_path(self, path, num_rows = None):
        """ take path of directory in which training images
            are stored and return pandas dataframe of features 
            
            args :
                path (str) : path of image folder
                num_rows (int) : if provided then return top num_rows rows from dataset (optional)
            
            Returns :
                Pandas DataFrame contained filled values
        """

        if not os.path.isdir(os.path.join(path)) :
            raise Exception('please provide directory path of training images')
        self._dataset = self._create_image_dataset(path)
        return self.dataset if num_rows is None else self.dataset.head(num_rows)


    def get_first_n_heterogeneity(self, n, data=None, seed = None):
        """ calculate heterogeneity till n
          
          args:
            n (int) : number of heterogeneity from k=1 to n
            data (numpy array) : data to which heterogeneity is getting calculated (optional)
            seed (int) : set seed to get constant randomness order (optional)
          
          Returns:
            Dictionary of k with heterogeneity
        """

        X = self._X if data is None else data
        if X is None :
            raise Exception('No data found')
        return self._km.get_initial_k(X, n=n, seed=seed)


    def get_initial_centroids_and_cluster_assignment(self, k=None, data=None, seed = None):
        """ initialize centroids and assignment through k mean algorithms
            
            args:
                k (int) ; number of clusters  (optional)
                data (numpy array) : numpy array of R, G, B value of each sample  (optional)
                seed (int) : set seed to get same randomness (optional)
                if argument is not provided value calculated on trining dataset
            
            Returns:
                centroids (numpy array) and cluster assignment (one hot encoded)
        """

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


    def get_em_params(self, data=None, maxiter = 1000, thresh=1e-4, seed = None):
        """ driver method to rum EM algorithms
            
            args :
                data (Pandas DatFrame): training dataset if not given then use internal dataset (optional)
                maxiter (int) : maximum iteration to rum EM (optional)
                threshold (float) : maximum threshold to stop EM (optional)
                seed (int) : set seed for randomness (optional)
            
            Returns :
                Dictionary contain clusters parameters i.e mean, weights, covariance, responsibility and number of iteration
        """

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


    def predict_soft_assignments(self, data, means=None, covariance=None, weight=None):
        """ predict the soft assignment of given data
            
            args :
                Data (numpy array) : data having R, G, B vectors for prediction
                means (numpy array) : mean co-ordinate of each clusters (optional)
                covariance metrix (numpy array) : covariance metrix (optional)
                weights (numpy array) : weight of each cluster (optional)
                if means, covariance and weight is not given then internal saved em output evaluated from dataset is used
            
            Returns :
                numpy array of responsibility
        """

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


    def get_first_n_data_responsibility(self, n, to_json=False):
        """ Get top n cluster wise responsibility from training data
            
            args :
                n (int) : number of responsibility return
                to_json (bool): if true then return in json format (optional)
            
            Returns :
                dictionary of cluster number to dataframe or json format responsibility and features detail
        """

        emobj = self
        df = emobj.dataset
        len_dataframe = len(df.index)
        if n > len_dataframe :
            raise Exception('required number of data is larger than available size of dataset')
        else :
            result = {}
            resp = pd.DataFrame(emobj.em_parameters['responsibility'], columns= list(range(emobj.k)))
            hard_assign = emobj.get_hard_assignment(to_pandas_DF=True)
            df = pd.concat([df, resp, hard_assign], axis=1) # concat base dataframe, responsibility and hard assignment Data Frame
            for i in range(emobj.k) :
                temp = df[df[self._assign_cluster_col_name] == i]
                temp = temp.sort_values(by=i, ascending=False) # sort by responsibility of ith cluster
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


    def predict_data(self, filename, filetype, filepath, val_base64, means=None, covariance=None, weight=None):
        """ predict the resposibility of given image path and
            return pandas dataframe contains features, responsibilities 
            and cluster assignment  value

            args :
                filename (str) :  file name
                filetype (str) : file extension
                filepath (str) : file full path
                val_base64 (str) : base 64 value of image in string
                means (numpy array) : mean of each cluster
                covariance (numpy array) : covariance metrix
                weight (numpy array) : weight of each clusters
            
            Returns :
                (Pandas DatFrame) of Image features and detail including responsibility
        """

        ext = '.' + filetype.split('/')[1]
        temp = self._get_image_feature(filepath)
        R, G, B = temp[0], temp[1], temp[2]
        resp = self.predict_soft_assignments(np.array([temp]), means=None, covariance=None, weight=None)
        hard_assign = self.get_hard_assignment(resp, self._k, to_pandas_DF=True)
        resp = pd.DataFrame(resp, columns= list(range(self.k)))
        df = self._get_base_dataset(filename, filepath, ext, R, G, B)
        df[self.base64_col_name] = val_base64
        df = pd.concat([df, resp, hard_assign], axis=1)
        df = self._shuffle_column(df)
        df = df.drop(['Path'], axis=1)
        return df



##########################################################################################################################

"""     Some Important Helper function are written Down
"""
##########################################################################################################################



def get_file_name(k, path, ext):
    """replace pickle file name non alphabatical character to '_' and then create hash value it
        
        args :
            path (str) : full path of file
            ext (str) : extension of file with .
        
        Returns :
            string with file Name seperated by '_'
    """
    file_name = re.sub('[^0-9a-zA-Z]+', '_', path) +'_' + str(k)
    hash_name = pkl.encode_file_name(file_name) # encode in sha to secure file Info
    return hash_name + ext


def get_k_mean_object():
    """ get K-mean object
        
        Returns :
            K_mean Object
    """

    return KM()


def get_EM_object():
    """get EM class object
        
        Returns :
            EM Object
    """

    return EM()


def get_EMBusiness_object(kmobj, emobj, train_data_dir_path, jsonpath):
    """
        get EMBusiness object with given parameter
        
        args :
            kmobj (object) : object of K_mean
            emobj (object) : object of EM
            train_data_dir_path (str) : training data folder full path
            jsonpath (str) : full json file path to store cluster number to name maping
        
        Returns :
            EMBusiness object
    """

    return EMBusiness(kmobj, emobj, train_dir_path=train_data_dir_path, cluster_name_json_file_path=jsonpath)


def write_em_pickle(k, train_data_dir_path, seed, f_name=None):
    """ parameter k-number of cluster, training path directory path, seed to
        initialize cluster parameter and f_name file name to save as pickle
        
        args :
            k (int) : number of clusters
            train_data_dir_path (str) : training data folder full path
            seed (int) : used to set randomness order for initialization of some parameters like initial centroids e.t.c
            f_name (string) : if provided then load only that pickle file (optional)
        
        Returns :
            written Pickled Object loaded from pickle file
    """

    f_name = f_name if f_name is not None else get_file_name(k, train_data_dir_path, '.pickle')
    kmobj = get_k_mean_object()
    emobj = get_EM_object()
    f_name_json = f_name.split('.')[0] + '.json' if f_name is not None else get_file_name(k, train_data_dir_path, '.json')
    jsonpath = pkl.get_json_file_path(f_name_json, CONST.EM_CLUSTER_NAME_SAVE_SUB_DIR)
    em_businessobj = get_EMBusiness_object(kmobj, emobj, train_data_dir_path, jsonpath)
    em_businessobj.k = k
    em_businessobj.get_em_params(seed=seed)
    pklobj = pkl.GetPickledObject.get_EM_pickled(train_data_dir_path, em_businessobj, k)
    pkl.write_pickled_object(pklobj, CONST.EM_PICKLES_SAVE_SUB_DIR, f_name)
    return pklobj


def get_em_object(k, train_data_dir_path, seed=None, saved_model_name=None):
    """ get pickled object from pickle file if not exist then
        create new pickle object then dump into file 
        
        args :
            k (int) : number of cluster
            train_data_dir_path (str) : training data folder full path (optional)
            seed (int) : used to set randomness order for initialization of some parameters like initial centroids e.t.c (optional)
            saved_model_name (string) : if provided then load only that pickle file model (optional)
        
        Returns :
            Pickled Object loaded from pickle file
    """

    f_name = saved_model_name if saved_model_name is not None else get_file_name(k, train_data_dir_path, '.pickle')
    pklobj = pkl.read_pickled_object(CONST.EM_PICKLES_SAVE_SUB_DIR, f_name)
    if pklobj is None : # If object not exist then create pickled file
        pklobj = write_em_pickle(k, train_data_dir_path, seed, f_name)
    return pklobj

