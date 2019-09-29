import os, base64
import pickle as pk
from pathlib import Path
from ML_algorithms.supporting_module  import pickle_module
from ML_algorithms.supporting_module.EMpickleclass import *
# Important constants used in this module
PATH = os.path.dirname(pickle_module.__file__)
MODEL_DIR = os.path.abspath(os.path.join(PATH, os.pardir)) + r'\Saved_Model' # directory path in which model pickled file is saved

# Interface class to get pickled object
class GetPickledObject :
    def __init__(self) :
        pass
    @staticmethod
    def get_EM_pickled(path, obj, k) :
        return EMPickledClass(path, obj, k)
# End
""" read model object from pickled file
input parameter pickled file path"""
def read_pickled_object(sub_dir, f_path) :
    try :
        pklobj = None
        path = MODEL_DIR + sub_dir
        if os.path.exists(os.path.join(path, f_path)) :
            with open(os.path.join(path, f_path), 'rb') as file:
                pklobj = pk.load(file)
        return pklobj
    except Exception as e:
        raise Exception(str(e))
# End
""" write pickled model object into pickled file
input parameter pickled object, pickled file path"""
def write_pickled_object(pkl_obj, sub_dir, f_name) :
    path = os.path.join(MODEL_DIR + sub_dir)
    # nested function
    def get_object() :
        with open(os.path.join(path, f_name), 'wb') as file:
            pk.dump(pkl_obj, file, protocol=pk.HIGHEST_PROTOCOL)
        return pkl_obj
    # End
    try :
        if os.path.exists(os.path.join(path)) :
            return get_object()
        else :
            #Path(os.path.join(path)).mkdir(parents=True, exist_ok=True)
            os.mkdir(path)
            return get_object()
    except Exception as e:
        raise Exception(str(e))
# End
""" encode file to base 64 encoding
imput parameter is file path"""
def encode_base64(path) :
    try :
        with open(os.path.join(path), "rb") as im_file:
            encoded_string = base64.b64encode(im_file.read())
        return encoded_string
    except :
        return None
# End
""" decode file from base 64 encoding
imput parameter is base64 string and file path"""
def decode_base64(base64, path) :
    try :
        img_str = base64.b64decode(base64)
        with open(os.path.join(path), 'wb') as f :
            f.write(img_str)
        return path
    except :
        return None
# End
