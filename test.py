import os
from ML_algorithms.cluster.EM import em_business as emb
TRAINING_PATH_DIR_EM = os.path.join(os.getcwd(), r'\Data\EM\images')
if __name__ == '__main__' :
    k=3
    #res = emb.get_first_n_heterogeneity(55)
    pkl_obj = emb.get_em_object(k, TRAINING_PATH_DIR_EM)
    if pkl_obj is None :
        raise Exception('no pickle object found')
    em_obj = pkl_obj.pickled_object
    data = em_obj._X[-2:]
    data = data if data.ndim == 2 else np.array([data]) if data.ndim == 1 else None 
    if data is not None :
        #res2 = em_obj.predict_soft_assignments(data, res['means'] ,res['covariances'], res['weights'])
        #res2 = em_obj.predict_soft_assignments(data)
        #print(res2)
        #print('original')
        #print(em_obj.em_parameters['responsibility'][-2:])
        res = em_obj.get_first_n_data_responsibility(5, em_obj)[0]
        for index, row in res.iterrows() :
            print(row['Image_Name'])