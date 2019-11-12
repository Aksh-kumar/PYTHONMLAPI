# PYTHONMLAPI
THis API contain Self Implemented expectation maximization(EM) algorithm in python using flesk framework to make its response available for consume by client side.THis Project structure is:

Folder structure start from base ~(PYTHONAPI)
 --->Data (for train data with respected implemented algorithms)
|     |
|     | 
|     -----> EM(train data from EM algorithms)
|             |
|             |
|             -------> images(contain the train set of images file used to train the EM algorithm)
|
|
---->Lib (Important packages used by flask framework)
|
|
---->ML_algorithms (open package for ML algorithms)
|         |
|         |
|         -----> Saved_Model (saved model objects)
|         |         |
|         |         |
|         |         -----> Cluster_Name_EM (saved Json cluster number to name mappint coming from client side)
|         |         |
|         |         |
|         |         -----> Pickle_EM (Contain pickle object for load pretrained model)
|         |
|         |-----> cluster (Clustering related algorithms)
|         |         |
|         |         |
|         |         ------>EM (Expectation Maximization Implementation)
|         |         |
|         |         |
|         |         ------> KM (k-mean Implementation)
|         |
|         ------> supporting_module (supporting module like pickle object used through out the package)
|         |
|         |
 ------> Scripts (flask related .dlls and .pyc files)
|
|
 ------> Temp (location for temporary file processing coming from client side)
|
|
------> main.py (flask API controller)
|
|
------> test.py (Not in use only for testing purpose)
      
 So this is my folder structure It might grow in future since I have Implemented Just a single algorithm.If you have any question regarding this then please contact me on resakash1498@gmail.com I am happy to answer.
