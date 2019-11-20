#!/usr/bin/env python
# coding: utf-8

import os
import base64
from hashlib import sha1
import pickle as pk
from pathlib import Path
from ML_algorithms.supporting_module  import pickle_module
from ML_algorithms.supporting_module.pickle_class import EMPickled
import ML_algorithms.constants as CONST

# constant created at run time to store file

# base path where pickle module reside

PATH = os.path.dirname(pickle_module.__file__)

# directory path in which model pickled file is saved

MODEL_DIR = os.path.abspath(os.path.join(PATH, os.pardir)) + CONST.SAVED_MODEL_FOLDER


class GetPickledObject:
    """
        Used as Interface class to get pickled object for different object
    """
    
    def __init__(self) :
        pass
    
    
    @staticmethod
    def get_EM_pickled(path, obj, k):
        """
            Used to get EMPickledClass object

            args :
                path (str) : training folder path used to train EM model
                obj (object) : EMBusiness class object that need to pickled
                k (int) : number of clusters used in training

            Returns :
                EMPickled object from pickled file
        """
        
        return EMPickled(path, obj, k)


def read_pickled_object(sub_dir, f_path):
    """
        read model object from pickled file
        
        args :
            sub_dir (str) : sub directory name contain inside saved_model
            f_path (str) : full file path of pickle file

        Returns :
            pickle class objects stored in file
    """
    
    try :
        pklobj = None
        path = MODEL_DIR + sub_dir
        if os.path.exists(os.path.join(path, f_path)) :
            with open(os.path.join(path, f_path), 'rb') as file:
                pklobj = pk.load(file)
        return pklobj
    except Exception as e:
        raise Exception(str(e))


def write_pickled_object(pklobj, sub_dir, f_name):
    """ write pickled model object into pickled file
        input parameter pickled object, pickled file path

        args :
            pklobj (object) : pickle object need to save
            sub_dir (string) : sub directory insede saved_model folder where need to save
            f_name (string) : file name to save file
        
        Returns :
            pickle object if successfully dump otherwise raise error
    """
    
    path = os.path.join(MODEL_DIR + sub_dir) # combine sub directory name to get full path
    
    def get_object():
        """
            nested function to load pickle object from file
        """
        
        with open(os.path.join(path, f_name), 'wb') as file:
            pk.dump(pklobj, file, protocol=pk.HIGHEST_PROTOCOL)
        return pklobj
    
    try :
        if os.path.exists(os.path.join(path)) :
            return get_object()
        else :
            os.mkdir(path) # If sub directory not exists then create
            return get_object()
    except Exception as e:
        raise Exception(str(e))


def encode_base64(path):
    """
        encode file to base 64 encoding
        input parameter is file path
        
        args :
            path (str) : path of file need to be encoded in base 64 format
        
        Returns :
            str : base 64 string
    """
    
    try :
        with open(os.path.join(path), "rb") as im_file:
            encoded_string = base64.b64encode(im_file.read())
        return encoded_string
    except :
        return None


def decode_base64(val_base64, path):
    """
        decode file from base 64 encoding input
        parameter is base64 string and file path
        
        args :
            val_base64 (str) : base 64 string
            path (str) : path of file where decoded base64 dumps
        
        Returns :
            (str) path of file if successfully decoded
    """
    
    try :
        img_str = base64.b64decode(val_base64)
        with open(os.path.join(path), 'wb') as f :
            f.write(img_str)
        return path
    except :
        return None


def get_json_file_path(filename, sub_dir):
    """
        get file path to save json file
        
        args :
            filename (str) : name of file
            sub_dir (str) : sub directory where json file is stored
        
        Returns :
            str full path string where Json file is stored under saved model
    """
    return os.path.join(MODEL_DIR, sub_dir, filename)


def encode_file_name(filename):
    """
        encode in sha1 for change filename before dumps
        
        args :
            filename (str) : file name need to hash
        
        Returns : 
            string of encoded hexdigits 
    """
    return sha1(filename.encode()).hexdigest()