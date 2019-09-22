import os, re, base64
import pickle as pk
from em_business import EMBusiness
import pandas as pd
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
    # replace file path non alphabatical character to _
    f_name =  re.sub('[^0-9a-zA-Z]+', '_', path) +'_' + str(k) + '.pickle'
    pklobj = None
    if os.path.exists(os.path.join(MODEL_DIR, f_name)) :
        with open(os.path.join(MODEL_DIR, f_name), 'rb') as file:
            pklobj = pk.load(file)
    else :
        em_obj = EMBusiness(path)
        em_obj.k = k
        em_obj.get_em_params(seed=SEED)
        pklobj = PickledClass(path, em_obj, k)
        with open(os.path.join(MODEL_DIR, f_name), 'wb') as file:
            pk.dump(pklobj, file, protocol=pk.HIGHEST_PROTOCOL)
    return pklobj
# End
def encode_base64(path) :
    try :
        with open(os.path.join(path), "rb") as im_file:
            encoded_string = base64.b64encode(im_file.read())
        return encoded_string
    except :
        return None
# End
def decode_base64(base64, path) :
    try :
        img_str = base64.b64decode(base64)
        with open(os.path.join(path), 'wb') as f :
            f.write(img_str)
        return path
    except :
        return None
# End
def get_first_n_data_responsibility(n, emobj, to_dict=False) :
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
            for index, row in temp.iterrows() :
                encd_64 = encode_base64(row['Path'])
                base64_list.append(encd_64)
            temp['Image_base64'] = base64_list
            result[i] = temp if not to_dict else temp.T.to_dict().values()
        return result
if __name__ == '__main__' :
    #pd.set_option('display.max_colwidth', -1)
    path = r'.\images'
    k = 3
    em_obj = get_EM_pickled_object(path, k).pickled_object
    #res = emb.get_first_n_heterogeneity(55)
    data = em_obj._X[-2:]
    data = data if data.ndim == 2 else np.array([data]) if data.ndim == 1 else None 
    if data is not None :
        #res2 = em_obj.predict_soft_assignments(data, res['means'] ,res['covariances'], res['weights'])
        #res2 = em_obj.predict_soft_assignments(data)
        #print(res2)
        #print('original')
        #print(em_obj.em_parameters['responsibility'][-2:])
        res = get_first_n_data_responsibility(5, em_obj)[0]
        for index, row in res.iterrows() :
            print(row['Image_Name'])