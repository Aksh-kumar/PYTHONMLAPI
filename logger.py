#!/usr/bin/env python
# coding: utf-8

import os
import datetime as dt
import application_constant as CONST

class LOGGER:
    """
    log any exception and write it into files
    """

    def __init__(self):
        pass

    
    @staticmethod
    def check_log_directory(log_dir_path):
        """
            check if log files directory present or not
            if not present then create the directory

            args :
                log_dir_path (str) : log files directory path

            Returns :
                bool true if exists false if exception raised while creations

        """
        
        try :
            
            if os.path.exists(log_dir_path) :
                return True
            else :
                os.mkdir(log_dir_path)
                return os.path.exists(log_dir_path)
        
        except NotImplementedError :
            
            return False


    @staticmethod
    def write_log_info(log_path, log_info) :
        """
            write log detail to file

            args :
                log_path (str) : folder path for log file
                log_info (str) : log details
        """
        
        filename = 'LOG_{datetime}.txt'.format(datetime=dt.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
        
        try :
            
            with open(os.path.join(log_path, filename), encoding='utf-8', mode='w+') as f : 
                f.write(log_info)
        
        except OSError :
            pass


    @staticmethod
    def LOG(log_info):
        """
            log all the exception detail into a file

            args :
                log_info (str) : Exception details
        """
        
        try :
            
            if LOGGER.check_log_directory(CONST.LOG_FILES_PATH):
                LOGGER.write_log_info(CONST.LOG_FILES_PATH, log_info)

        except :
            pass # If exception raised at last levels