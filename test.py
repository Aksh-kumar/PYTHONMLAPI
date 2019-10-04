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
import json
with open('check_res.json') as f:
    data = json.load(f)
#data = dic(data)
if __name__ == '__main__' :
    #res = emb.get_first_n_heterogeneity(55)
    #print(TRAINING_PATH_DIR_EM)
    #assert False
    k = 4
    pkl_obj = emb.get_em_object(k, TRAINING_PATH_DIR_EM, seed=SEED)
    if pkl_obj is None :
        raise Exception('no pickle object found')
    em_obj = pkl_obj.pickled_object
    #img_base64 = data['value']
    #img_name = data['filename']
    #iletype = data['filetype']
    #k = data['k']
    #path = os.path.join(TEMP_FILE_PATH, img_name)
    #path = spr.decode_base64(img_base64, path)
	#print(path)
    #if path is not None :
	# emobj = get_model(k)
	#print(emobj.predict_data(img_name, filetype, path, img_base64).T.to_dict())
    # get first n heteroginity
    #print(em_obj.get_first_n_heterogeneity(55, seed=SEED))
    #data = em_obj._X[-2:]
    #data = data if data.ndim == 2 else np.array([data]) if data.ndim == 1 else None 
    #if data is not None :
    #res2 = em_obj.predict_soft_assignments(data, res['means'] ,res['covariances'], res['weights'])
    #res2 = em_obj.predict_soft_assignments(data)
    #print(res2)
    #print('original')
    #print(em_obj.em_parameters['responsibility'][-2:])
    # first n responsibility
    res = em_obj.get_first_n_data_responsibility(5, to_json=True)
    for i in res.keys() :
        res[i] = ast.literal_eval(res[i])
    print(res) 
    # em paramter
    #print(em_obj.em_parameters)
    # image extension supported
    #print(em_obj.IMAGE_EXT_SUPPORTED)
    # get cluster name
    #print(em_obj.cluster_name)
    # set cluster name
    #if em_obj.k == len(dic.keys())
    #em_obj.cluster_name = dic
    
    #for index, row in res.iterrows() :
    #print(row[])