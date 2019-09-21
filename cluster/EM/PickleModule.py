import os, re
import pickle as pk
from em_business import EMBusiness
# Important constants used in this module
MODEL_DIR = r'.\Saved_Model' # directory path in which model pickled file is saved
SEED = 0 # seed used to initiate centroids in k-mean algorithm
class PickledClass :
    """Used to save Model with meta details like path of training data directory
    number of clusters and model object"""
    def __init__(self, path, obj, k) :
        self._path = path
        self._obj = obj
        self._k = k
    @property
    def path(self) :
        """ return training path of contained model object"""
        return self._path
    @property
    def pickled_object(self) :
        """ returned model object """
        return self._obj
    @property
    def k(self) :
        """ return number of clusters in model objects"""
        return self._k
# End

""" returned model object from pickled file
if not present then create model object and
then pickled it and return that model object
Input parameter training path directory and
number of cluster"""
def get_EM_pickled_object(path, k) :
    global MODEL_DIR
    global SEED
    f_name =  re.sub('[^0-9a-zA-Z]+', '_', path) +'_' + str(k) + '.pickle'
    em_obj = None
    if os.path.exists(os.path.join(MODEL_DIR, f_name)) :
        with open(os.path.join(MODEL_DIR, f_name), 'rb') as file:
            pklobj = pk.load(file)
            em_obj = pklobj.pickled_object
        res = em_obj.em_parameters
    else :
        em_obj = EMBusiness(path)
        em_obj.k = k
        em_obj.get_em_params(seed=SEED)
        pklobj = PickledClass(path, em_obj, k)
        with open(os.path.join(MODEL_DIR, f_name), 'wb') as file:
            pk.dump(pklobj, file, protocol=pk.HIGHEST_PROTOCOL)
    return em_obj
# End

if __name__ == '__main__' : 
    path = r'.\images'
    k = 3
    em_obj = get_EM_pickled_object(path, k)
    #res = emb.get_first_n_heterogeneity(55)
    data = em_obj._X[-2:]
    data = data if data.ndim == 2 else np.array([data]) if data.ndim == 1 else None 
    if data is not None :
        #res2 = em_obj.predict_soft_assignments(data, res['means'] ,res['covariances'], res['weights'])
        res2 = em_obj.predict_soft_assignments(data)
        print(res2)
        print('original')
        print(res['responsibility'][-2:])