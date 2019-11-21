# PYTHONMLAPI
THis API contain Self Implemented expectation maximization(EM) algorithm in python using flesk framework to make its response available for consume by client side. the client side code written in angular 8 available in https://github.com/Aksh-kumar/ML_AlgorithmUI. A pretrained model is already been saved for demo purposes based on data located in ~/Data/EM/images/ to load this model just make 
USED_SAVED_MODEL_FOR_4_CLUSTER to True in application_constant.py file. THis project is fully customize to use. If you have different training data then put that data inside data folder and update TRAINING_PATH_DIR_EM variable in application_constant.py file. ML_algorithms package source code is open intentionally for modification so that it can be used in different project, right now it contain only EM and K-mean algorithm but in future it will grow. This Project structure is:

## Folder structure start from base ~(PYTHONAPI)
<pre>
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
----> Log (contain log information files if some exception is raised) 
|
|
---->ML_algorithms (open package for ML algorithms)
|         |
|         |
|         -----> saved_model (saved model objects)
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
|         |         ------>Expectation_Maximization (Expectation Maximization Implementation folder)
|         |         |
|         |         |
|         |         ------> K_mean (k-mean Implementation folder)
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
------> application_constant.py (contain application related constant variable)
|
|
------> logger.py (contain logger class to log Exception details)
|
|
------> test.py (Not in use only for testing purpose)
|
|
-----> pyvenv.cfg(configuration detail related to python path)
</pre>

 change pyvenv.cfg home and version to your installed python for environment.
 
 ## important library used :
 ### imageio (for Image file processing)
 ### pandas (for handling dataset)
 ### numpy and scipy (for mathematical operations)
 ### pickle (for save the model)
 ### flask (for REST API communications)
 
 If you have any question regarding this then please contact me on resakash1498@gmail.com I am happy to answer.
