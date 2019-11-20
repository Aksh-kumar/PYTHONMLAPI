#!/usr/bin/env python
# coding: utf-8

import os
import sys
import traceback as tb
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
        
        except NotImplementedError:
            return False


    @staticmethod
    def write_log_info(log_path, exceptionobj) :
        """
            write log detail to file

            args :
                log_path (str) : folder path for log file
                exceptionobj (object) : Exception object
        """
        
        filename = 'log_{datetime}.txt'.format(datetime=dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        
        try :
            with open(os.path.join(log_path, filename), encoding='utf-8', mode='w') as f : 
                f.write(''.join(tb.format_tb((exceptionobj.__traceback__))))
        
        except OSError:
            pass


    @staticmethod
    def LOG(exceptionobj, trace_back_info=sys.exc_info()):
        """
            log all the exception detail into a file

            args :
                exceptionobj (object) : Exception object
        """
        
        try :
            
            if LOGGER.check_log_directory(CONST.LOG_FILES_PATH):
                LOGGER.write_log_info(CONST.LOG_FILES_PATH, exceptionobj)

        except Exception:
            pass # If exception raised at last levels