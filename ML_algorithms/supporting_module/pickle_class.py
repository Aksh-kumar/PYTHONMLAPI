#!/usr/bin/env python
# coding: utf-8

class EMPickled:
    """
        Used to preserve EMBusiness object as pickle 
        file and some meta data like training files
        path and value of k which can be later on used
    """
    
    def __init__(self, path, obj, k):
        """
            Initialize all variables
            
            args :
                path (str) : training folder path
                obj (object) : EmBusiness class object
                k (int) : number of cluster used
        """
        self._path = path
        self._obj = obj
        self._k = k
    
    
    @property
    def path(self):
        """
            Get training path of EMBusiness model object

            Returns :
                string training folder path sed to train the model
        """
        return self._path
    
    
    @property
    def pickled_object(self):
        """
            Get EMBusiness class model object 
            
            Returns :
                EMBusiness object
        """
        return self._obj
    
    
    @property
    def k(self) :
        """
            Get number of clusters in model objects

            Returns :
                int number of cluster used
        """
        return self._k
