import os

CURRENT_DIR = os.getcwd() # save corrent directory path

TEMP_FILE_PATH = os.path.join(CURRENT_DIR, r'Temp') # store temporary files used to process

SEED = 0 # used to set the constant random orders

# EM related constants

USED_SAVED_MODEL_FOR_4_CLUSTER = True # make it false if don't want to use saved model

TRAINING_PATH_DIR_EM = os.path.join(CURRENT_DIR, r'Data\EM\images') # training data 

SAVED_MODEL_NAME = 'saved_4_model.pickle' # name of 4 clusterd saved model effective when USED_SAVED_MODEL_FOR_4_CLUSTER = True