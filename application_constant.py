import os

################################### main module constants ####################################################################

CURRENT_DIR = os.getcwd() # save current directory path

TEMP_FILE_PATH = os.path.join(CURRENT_DIR, r'Temp') # store temporary files used to process

LOG_FILES_PATH = os.path.join(CURRENT_DIR, r'Log') # store log files

SEED = 0 # used to set the constant random orders

################################################################################################################################

################################### EM  constants  #############################################################################

USED_SAVED_MODEL_FOR_4_CLUSTER = False # make it True if want to use saved model

TRAINING_PATH_DIR_EM = os.path.join(CURRENT_DIR, r'Data\EM\images') # training data 

SAVED_MODEL_NAME = 'saved_4_model.pickle' # name of 4 clusterd saved model effective when USED_SAVED_MODEL_FOR_4_CLUSTER = True

################################################################################################################################