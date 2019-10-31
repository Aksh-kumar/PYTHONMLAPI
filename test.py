import os, json, ast
import numpy as np
from ML_algorithms.cluster.EM import em_business as emb
from ML_algorithms.supporting_module import pickle_module as spr
SEED = 0
# INIT Important class
TRAINING_PATH_DIR_EM = os.path.join(os.getcwd(), r'Data\EM\images')
CURRENT_DIR = os.getcwd()
TEMP_FILE_PATH = os.path.join(CURRENT_DIR, r'Temp')
dic_model = {}
def get_model(k) :
	global TRAINING_PATH_DIR_EM
	global SEED
	global dic_model
	if k in dic_model :
		return dic_model[k]
	pklobj = emb.get_em_object(k, TRAINING_PATH_DIR_EM, seed=SEED)
	if pklobj is None :
		raise Exception('no pickle object found')
	emobj = pklobj.pickled_object
	dic_model[k] = emobj
	return emobj
# End
if __name__ == '__main__' :
    k = 4
    pkl_obj = emb.get_em_object(k, TRAINING_PATH_DIR_EM, seed=SEED)
    if pkl_obj is None :
        raise Exception('no pickle object found')
    em_obj = pkl_obj.pickled_object
    # first n responsibility
    res = em_obj.get_first_n_data_responsibility(5, to_json=True)
    for i in res.keys() :
        res[i] = ast.literal_eval(res[i])
    print(res) 
    