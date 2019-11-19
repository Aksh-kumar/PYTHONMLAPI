#!/usr/bin/env python
# coding: utf-8

# constant used By EM and EM Business classes
EM_PICKLES_SAVE_SUB_DIR = r'\pickled_EM' # saved sub directory of pickle file

EM_CLUSTER_NAME_SAVE_SUB_DIR = r'cluster_name_EM' # location where cluster number to name pmapped son is saved

SUPPORTED_IMAGES_EXTENSION = ['.png', '.jpeg', '.jpg', '.jif', '.jpe'] # Image file extension supported

DATASET_COL = ['ImageName', 'Path', 'Extension', 'R', 'G', 'B' ] # Order of pandas dataset created inside business class

COL_BASE64_NAME = 'ImageBase64' # column name contain image base 64 encoded string

COL_ASSIGN_CLUSTER = 'AssignCluster' # column name contain cluster number of image assigned

# saved pickle file folder name

SAVED_MODEL_FOLDER = r'\saved_model'  # saved_model folder name
