import os
import sys
from logging import (DEBUG, INFO, WARNING, ERROR, FileHandler, Formatter, StreamHandler, getLogger)
from array import *
from utils import *

class Handler():
    def __init__(self, name='handler', verbosity='debug', basepath=None, overwrite=False, tensorboard=False):
        '''
            Intializes the Logger / Handler class. 
            This class is a helper for model.py and to log messages and create paths in a consistent way.
            Args: 
                name: the name of the handler for logging, etc. - will also create corresponding folders 
                verbosity: the verbosity for logging ('debug', 'info', 'error' or 'warning')
                basepath: (string) absolute path (including last slash) to where results should be stored, if None - will be created based on GIT-structure
                overwrite: Boolean to set to True if already existing files and paths should be deleted, when a handler with the same name and basepath already existed
                tensorboard: Boolean to indicate if also subfolders for tensorboard logs should be created.
        '''
        self.name = name 
        self.set_verbosity(verbosity)

        self.overwrite = overwrite
        self.tensorboard = tensorboard

        # Basepath creation 
        if basepath is None: 
            self.basepath = self.create_basepath()
            if self.tensorboard: 
                self.tensorboard_path = self.create_tensorboardpath()
        else: 
            assert isinstance(basepath, str), 'Handler Class: Basepath can only of type string or None!'
            if not basepath.endswith('/'): 
                basepath += '/'
            
            self.basepath = basepath + self.name + '/'
            if self.tensorboard:
                self.tensorboard_path = basepath + 'tensorboards/{}/'.format(self.name)
        
        if self.overwrite: 
            self.delete_existing_files()

        mkdir(self.basepath)
        if self.tensorboard:
            mkdir(self.tensorboard_path)

        # Logger Creation 
        self.logger = None
        self.setup_logger()
        
        self.log('debug', 'Handler: Intialization successful. Name: {}'.format(self.name))

    def get_verbosity(self): 
        '''
            Returns the current verbosity setting.
        '''
        return self.verbosity

    def set_verbosity(self, verbosity):
        '''
            Sets the verbosity level of the Handler. 
            Args: 
                verbosity: string parameter that can be one of the following: debug, info, error, warning
        ''' 
        
        assert str.lower(verbosity) in ['debug', 'info', 'error', 'warning'], 'Handler: Verbosity [{}] is not defined - only use one of the following [info, debug, error, warning].'.format(verbosity)
        self.verbosity = str.lower(verbosity)

    def delete_existing_files(self): 
        '''
            Deletes the files and subfolders in self.basepath if they do exist.
        '''
        
        os.system('rm -rf {}'.format(self.basepath))
        if self.tensorboard:
            os.system('rm -rf {}'.format(self.tensorboard_path))

    def create_tensorboardpath(self): 
        '''
            Creates the path to store the tensorboard files (one folder up from basebath and in two subfolders)
        '''
        
        direct = os.path.dirname(self.basepath)
        direct = os.path.dirname(direct)
        direct += '/tensorboards/{}/'.format(self.name)
        return direct

    def create_basepath(self): 
        '''
            Create the basepath of the handler based on this GIT-structure and where the code is located.
        '''
        direct = os.path.dirname(os.path.abspath(__file__))
        direct = os.path.dirname(direct)
        direct += '/results/{}/'.format(self.name)
        return direct

    def setup_logger(self):
        '''
            Sets up the logger to later use it for any kind of logging. 
        '''
        if self.verbosity is None:
            self.logger = None
            return
        self.logger = getLogger(self.name)
        self.logger.setLevel(DEBUG)
        self.logger.propagate = False
        if not self.logger.handlers:
            fh = FileHandler(os.path.join(self.basepath, 'logging.log'))
            fh.setLevel(DEBUG)
            ch = StreamHandler(sys.stdout)
            ch_level = {'info': INFO, 'debug': DEBUG, 'error' : ERROR, 'warning' : WARNING}[self.verbosity]
            ch.setLevel(ch_level)
            
            formatter = Formatter('[%(asctime)s {}] %(message)s'.format(self.name))
            formatter.datefmt = '%y/%m/%d %H:%M:%S'
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def log(self, logtype, msg):
        '''
            Logs messages. 
            Args: 
                logtype: the type of logging message - can be one of the following: ['debug', 'info', 'warning', 'error']
                msg: The string to log
        '''
        if self.logger is None:
            return

        if logtype == 'debug':
            self.logger.debug('[DEBUG] {}'.format(msg))
        elif logtype == 'info':
            self.logger.info('[INFO] {}'.format(msg))
        elif logtype == 'warning':
            self.logger.warning('[WARNING] {}'.format(msg))
        elif logtype == 'error':
            self.logger.error('[ERROR] {}'.format(msg))

    def path(self, path=None, use_results_path=False, no_mkdir=False):
        '''
            Creates a path in sense of the Handler.
            This means that the Handler has a basepath where it is working on (e.g logging).
            The path will be 'concatenated' to the basepath and will create corresponding subfolders if they do not exist yet. 
            Args: 
                path: the path you want to adapt in the Logger's sense
                use_results_path: Boolean to set to True if a folder should not be created in the experiment folder, but above of that in the results folder
                no_mkdir: Boolean to set True if path should only be returned but folders should not be created in case of no existence
            Returns: 
                the 'concatenated' path in the handler folders

        '''
        if not use_results_path: 
            tp = self.basepath
        else: 
            tp = self.results_path
        if path is not None:
            tp = os.path.join(tp, path)
        dirpath = tp 
        if '.' in os.path.split(tp)[1]:
            dirpath = os.path.split(tp)[0]
        if not no_mkdir: 
            mkdir(dirpath)
        return tp + '/'

    def __repr__(self): 
        '''
            Overwrite string representation of Handler object.
        '''
        s = '-------------------------\n'
        s += 'Class: Handler\n'
        s += '-------------------------\n'
        s += 'Name: {}\n'.format(self.name)
        s += 'Tensorboard Activated: {}\n'.format(self.tensorboard)
        s += 'Base-Path: {}\n'.format(self.basepath)
        s += 'Tensorboard-Path: {}\n'.format(self.tensorboard_path)
        s += 'Verbosity: {}\n'.format(self.verbosity)
        s += 'Overwrite: {}\n'.format(self.overwrite)
        s += '-------------------------\n'
        return s

if __name__ == '__main__': 
    handler = Handler(name='test', verbosity='debug', overwrite=True, tensorboard=True)
    print(handler)


