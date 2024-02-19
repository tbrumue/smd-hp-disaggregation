from datetime import datetime
from multiprocessing import cpu_count
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random as rn
import os
import numpy as np
import pickle

# set device visibility for tensorflow
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

# import own code 
from handler import *
from utils import *
from resultreporter import *

# import tensorflow and keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.backend import manual_variable_initialization 
manual_variable_initialization(True)
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# set random seeds 
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
rn.seed(0)
tf.random.set_seed(0)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

class Model():

    def __init__(self, 
                 model_name:str, 
                 fold:int=None,
                 layer_shapes:list=[50, 100, 200, 100, 50], 
                 epochs:int=200, 
                 batch_size:int=2048,
                 lr_init:float=0.001, 
                 lr_schedule:str='constant',
                 beta_1:float=0.9, 
                 l2_regularization_weight:float=None, 
                 dropout:float=None, 
                 early_stopping_patience:int=10,
                 scale_X:bool=True, 
                 scale_y:bool=False,  
                 basepath:str=None, 
                 suppress_logs:bool=False, 
                 apple_silicon:bool=False):
        '''
            Class for fully-connected neural network for sequence-to-sequence energy consumption prediction.
        
            Args:
                model_name (str): Custom name of the model - if None: a unique name is created based on the parameters of the model (requires fold to be set as well)
                fold (str): Number of current fold of k-fold crossvalidation - NOTE: only used for model_name creation when no custom model name is provided
                layer_shapes (list): List of number of neurons per layer - length of the list defines the number of intermediate layers, e.g. [50, 100, 200, 100, 50] creates a network with 5 layers
                epochs (int): Number of trained epochs - NOTE: this parameter is only used during training 
                batch_size (int): Size of used minibatches during training - NOTE: this parameter is only used during training 
                lr_schedule (str): If "constant": learning_rate_init is used for whole training - if "exponential": learning rate is reduced after 10 epochs
                lr_init (str): Parameter of Adam optimizer. Defaults to 0.001 - NOTE: this parameter is only used during training 
                beta_1 (float): Parameter of Adam optimizer. Defaults to 0.9 - NOTE: this parameter is only used during training 
                l2_regularization_weight (float): if None, no weight regularization is used during training - otherwise this parameter defines the weight of the L2 regularization - NOTE: this parameter is only used during training 
                dropout (float): Whether to use dropout as regularization technique and if yes, this is the probability. Defaults to None - NOTE: this parameter is only used during training
                early_stopping_patience (int): Number of epochs to wait for improvement before stopping training. Defaults to 10 - NOTE: this parameter is only used during training
                scale_X (bool): Decides if input data is scaled. Defaults to True - NOTE: scaler created during training or by loading from file
                scale_y (bool): Decides if output data is scaled. Defaults to False - NOTE: scaler created during training or by loading from file
                basepath (string): optional string to location where subfolders should be created by the handler object - When set to None - the handler object will create a basepath based on the GIT-structure
                suppress_logs (bool): Boolean to set to True when print outs should be suppressed - NOTE: this will also affect the logs. No logs will be made! Defaults to False.
                apple_silicon (bool): Decides if Apple Silicon is used. Defaults to False. If True, a legacy version of the Adam optimizer is used for training.
        '''

        assert lr_schedule in ['constant', 'exponential'], 'Model.__init__(): Currently only the following values are supported for the lr_schedule parameter: constant, exponential.'
        assert isinstance(l2_regularization_weight, float) or l2_regularization_weight is None, 'Model.__init__(): Parameter l2_regularization_weight must be a float or None. Current type: {}'.format(type(l2_regularization_weight))
        if isinstance(l2_regularization_weight, float):
            assert l2_regularization_weight > 0.0 and l2_regularization_weight < 1.0, 'Model.__init__(): Parameter l2_regularization_weight must be a float between 0.0 and 1.0. Current value: {}'.format(l2_regularization_weight) 
        assert model_name is None or isinstance(model_name, str), 'Model.__init__(): Parameter model_name must be a string or None.'
        if model_name is None:
            assert isinstance(fold, int), 'Model.__init__(): Parameter fold must be an integer. Current value: {}'.format(fold)
        assert isinstance(layer_shapes, list) and all([isinstance(e, int) for e in layer_shapes]), 'Model.__init__(): Parameter layer_shapes must be a list of integers.'
        assert isinstance(lr_init, float) and lr_init > 0, 'Model.__init__(): Parameter lr_init must be a float greater than 0.'
        assert isinstance(beta_1, float) and beta_1 > 0, 'Model.__init__(): Parameter beta_1 must be a float greater than 0.'
        assert isinstance(dropout, float) or dropout is None, 'Model.__init__(): Parameter dropout must be a float between 0 and 1 or None.'
        if isinstance(dropout, float):
            assert dropout > 0 and dropout < 1, 'Model.__init__(): Parameter dropout must be a float between 0 and 1 or None.'
        assert isinstance(scale_X, bool), 'Model.__init__(): Parameter scale_X must be a boolean.'
        assert isinstance(scale_y, bool), 'Model.__init__(): Parameter scale_y must be a boolean.'
        assert isinstance(apple_silicon, bool), 'Model.__init__(): Parameter apple_silicon must be a boolean.'
        assert basepath is None or isinstance(basepath, str), 'Model.__init__(): Parameter basepath must be a string or None.'
        assert isinstance(suppress_logs, bool), 'Model.__init__(): Parameter suppress_logs must be a boolean.'
        assert isinstance(early_stopping_patience, int) or early_stopping_patience is None, 'Model.__init__(): Parameter early_stopping_patience must be an integer or None.'
        if isinstance(early_stopping_patience, int):
            assert early_stopping_patience > 0, 'Model.__init__(): Parameter early_stopping_patience must be an integer greater than 0.'

        self.fold = fold
        self.suppress_logs = suppress_logs
        if model_name is None:
            self.model_name = create_unique_model_name(layer_shapes, epochs, batch_size, lr_init, beta_1, fold)
        else: 
            self.model_name = model_name

        self.handler = Handler(name=self.model_name, basepath=basepath, tensorboard=True, overwrite=True) # overwrite Handler object
        self.tensorboard_path = self.handler.tensorboard_path + self.model_name # + datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + '_' + str(self.batch_size) + '_' + str(self.fold)
        
        if l2_regularization_weight is None:
            self.regularizer = None
        else:
            self.regularizer=keras.regularizers.l2(l2_regularization_weight)
        
        self.dropout = dropout    
        self.apple_silicon = apple_silicon  
        self.scale_X = scale_X
        self.scale_y = scale_y
        self.layer_shapes = layer_shapes
        self.epochs = None
        self.set_epochs(epochs)
        self.batch_size = None
        self.set_batch_size(batch_size)
        self.lr_init = lr_init
        self.lr_schedule = lr_schedule
        self.beta_1 = beta_1
        self.early_stopping_patience = early_stopping_patience
        self.model_initialized = False # will be set to True when __init_nn__() is called, which is the case when starting training for the first time 

        self.Xscaler = None # will be filled when calling fit_transform_X_scaler() or when set_X_scaler() is called
        self.Yscaler = None # will be filled when calling fit_transform_y_scaler() or when set_y_scaler() is called

        self.model = None # holds actual keras object will be filled during __init_nn__() call or when loading a model 
        self.input_dim = None # NOTE: will be filled when fit() is called for the first time - based on the data provided
        self.output_dim = None # NOTE: will be filled when fit() is called for the first time - based on the data provided

        self.checkpoint_callback = None # will be filled during create_callbacks() call
        self.tensorboard_callback = None # will be filled during create_callbacks() call
        self.early_stopping_callback = None # will be filled during create_callbacks() call
        self.csv_logger_callback = None # will be filled during create_callbacks() call

        self.__log__('Model.__init__(): Object creation successful.', level='debug')


    def __init_nn__(self):
        '''
            Initialize neural network model with given parameters.
            Will be called during trainig process.
        '''

        if not self.model_initialized: 

            # create callbacks for tensorboard, filewriting, early stopping, learning rade scheduling and checkpointing
            self.create_callbacks()

            # create model object
            self.model = keras.Sequential()

            # create input layer 
            self.model.add(keras.Input(shape=(self.input_dim,)))

            # create intermediate layer with given size of nodes for each entry in layer shapes
            for index, layer in enumerate(self.layer_shapes):
                self.model.add(keras.layers.Dense(layer, activation='relu', name = 'layer'+str(index)))
                
                # optionally add dropout layer with previously defined probability
                if self.dropout is not None:
                    self.model.add(keras.layers.Dropout(self.dropout))

            # add output layer  
            # NOTE: RELU is used as activation function for the output layer - this is not standard but was used in the paper because energy consumption cannot be negative
            self.model.add(keras.layers.Dense(self.output_dim, activation='relu', name = 'output_layer', kernel_regularizer=self.regularizer))
            self.model.add(keras.layers.ReLU())
            
            # build and compile model 
            self.model.build()
            if self.apple_silicon: # for Apple silicon chips it is currently recommended to use the legacy version of the Adam optimizer
                optimizer=keras.optimizers.legacy.Adam(learning_rate=self.lr_init, beta_1=self.beta_1)
            else: 
                optimizer = keras.optimizers.Adam(learning_rate=self.lr_init, beta_1=self.beta_1)

            # compile model - NOTE: we use mean squared error as loss function, RMSE and MAE as metrics (as defined in the paper)
            self.model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError()])
            
            # set flag to True to indicate that model has been initialized
            self.model_initialized = True            
            self.__log__('Model.__init_nn__(): Initialization successful.', level='debug')

        elif self.model.checkpoint_callback is None or self.tensorboard_callback is None:
            self.__log__('Model.__init_nn__(): Model already initialized but checkpoint callback is missing. Creating callbacks.', level='debug')
            self.create_callbacks()
        else: 
            self.__log__('Model.__init_nn__(): Model already initialized. Skipping initialization.', level='debug')
        
    def create_callbacks(self):
        '''
            Create callbacks for training process for tensorboard, filewriting, early stopping, learning rade scheduling and checkpointing.
        '''
        self.checkpoint_callback = keras.callbacks.ModelCheckpoint(self.handler.path('model'), monitor='val_loss', verbose=0, save_best_only=True, mode='min')
        self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.tensorboard_path, write_images=True, histogram_freq=1)
        if self.early_stopping_patience is not None:
            self.early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.early_stopping_patience) # restore_best_weights=True
        self.lr_callback = keras.callbacks.LearningRateScheduler(self.lr_scheduler)
        self.csv_logger_callback = keras.callbacks.CSVLogger(self.handler.path('training_scores.csv'), append=True, separator=';')
        self.__log__('Model.create_callbacks(): Callbacks created.', level='debug')
        
    def lr_scheduler(self, epoch, lr):
        """Creates a learning rate schedule. This function can be called during the keras training process. Not used in my Thesis.

        Args:
            epoch (int): Training epoch
            lr (float): Current learning rate

        Returns:
            float: new learning rate
        """
        if self.lr_schedule == 'constant':
            return lr
        elif self.lr_schedule == 'exponential':
            if epoch < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.1)

    def fit(self, X, y, validation_split:float=0.1, shuffle_data:bool=True, verbose:int=2): 
        '''
            Fit model with given data.
            NOTE: This function is the main entry point for training the model. It will call all necessary functions to prepare the data and the model for training.
            NOTE: when reloading a model with load_keras_model() and retraining it with this function, you may want to adjust the batchsize and epoch with set_batch_size() and set_epochs().

            1. Scale input
            2. Scale output
            3. Shuffle training data to prevent overfitting to last househoulds in training data.
            4. Split training data into training and validation
            5. Fit model
            6. Save model

            Args: 
                X: input data
                y: output data
                validation_split: float value between 0 and 1 to split the data into training and validation data
                shuffle_data: boolean to decide if data should be shuffled before training
                verbose: integer to decide how much information should be printed during training (0 = nothing, 1 = progress bar, 2 = full information)
        '''     

        assert X.shape[0] == y.shape[0], 'Model.fit(): X and y do not match in shape. X.shape[0] = {}, y.shape[0] = {}.'.format(X.shape[0], y.shape[0])
        assert isinstance(validation_split, float) and validation_split > 0 and validation_split < 1, 'Model.fit(): validation_split has to be a float value between 0 and 1. Current value: {}'.format(validation_split)
        assert isinstance(shuffle_data, bool), 'Model.fit(): shuffle_data has to be a boolean. Current value: {}'.format(shuffle_data)
        assert isinstance(verbose, int) and verbose >= 0 and verbose <= 2, 'Model.fit(): verbose has to be an integer between 0 and 2. Current value: {}'.format(verbose)
        assert self.batch_size is not None, 'Model.fit(): Batch size has to be set before training. Please use set_batch_size() to set the batch size.'
        assert self.epochs is not None, 'Model.fit(): Number of epochs has to be set before training. Please use set_epochs() to set the number of epochs.'

        self.__log__('Model.fit(): Starting training.', level='info')

        # preparations 
        # start_time = datetime.now() # currently not used - but can be used to measure computing time
        keras.backend.clear_session()

        # set dimensions for the input and output layer - will be used during initialization of the model
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        else: 
            assert self.input_dim == X.shape[1], 'Model.fit(): Input dimension of model does not match to the input data. Model.input_dim = {}, X.shape[1] = {}.'.format(self.input_dim, X.shape[1])
        
        if self.output_dim is None:
            self.output_dim = y.shape[1]
        else:
            assert self.output_dim == y.shape[1], 'Model.fit(): Output dimension of model does not match to the output data. Model.output_dim = {}, y.shape[1] = {}.'.format(self.output_dim, y.shape[1])
        
        # initialize model - NOTE: if the model has been initialized already, the intitialization will be skipped within the function
        self.__init_nn__()
        
        X_train = X.copy()
        y_train = y.copy()

        # 1. Scale input
        if self.scale_X:
            X_train = self.fit_transform_X_scaler(X_train)

        # 2. Scale output
        if self.scale_y:
            y_train = self.fit_transform_y_scaler(y_train)

        # 3. Shuffle data
        if shuffle_data:
            X_train, y_train = shuffle(X_train, y_train)

        # Split data into training and validation 
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=0)

        # get the callbacks - will be filled during __init_nn__() call
        if self.early_stopping_callback is not None: 
            callbacks = [self.tensorboard_callback, self.checkpoint_callback, self.early_stopping_callback, self.lr_callback, self.csv_logger_callback]
        else: 
            callbacks = [self.tensorboard_callback, self.checkpoint_callback, self.lr_callback, self.csv_logger_callback]

        # 4. fit model and save
        # NOTE: all CPU cores used with cpu_count() and use_multiprocessing=True - adjust for your own use case
        # NOTE: you may also want to optimize it for GPU usage 
        self.model.fit(X_train, y_train, verbose=verbose, epochs=self.epochs, batch_size=self.batch_size, workers=cpu_count(), use_multiprocessing=True, callbacks=callbacks, validation_data=(X_val, y_val))

        # load best model from check point if it has already been check pointed otherwise keep it as it is 
        try:
            self.model = keras.models.load_model(self.handler.path('model'))
            # self.model.load_weights(self.handler.path('model'))
            self.__log__('Model.fit(): Reloaded best model from checkpoint.', level='debug')
        except: 
            self.__log__('Model.fit(): Unable to reload best model from checkpoint. Using latest model instead.', level='error')
            pass
        
        # save X and y scaler
        if self.scale_X:
            self.save_x_scaler()
        if self.scale_y:
            self.save_y_scaler()

        self.__log__('Model.fit(): Finished training.', level='info') 
        
        # NOTE: computing time is not used at the moment but can be calculated as follows if needed
        # end_time = datetime.now()
        # computing_time = str(end_time - start_time)

    def predict(self, X, verbose=0):
        '''
            Pipeline to make a prediction.
            1. Scale input
            2. reverse transform output in case a scaler is used.

            Args:
                X (numpy.ndarray): Test data
                verbose (int): Verbosity level of the prediction process (0 = nothing, 1 = progress bar, 2 = full information)
            Returns:
                prediction (numpy.ndarray): data predicted by model
        '''
        assert X.shape[1] == self.input_dim, 'Model.predict(): Shape of data provided does not match to the input dimension of the model. X.shape[1] = {}, input_dim = {}.'.format(X.shape[1], self.input_dim)
        assert isinstance(verbose, int) and verbose >= 0 and verbose <= 2, 'Model.predict(): verbose has to be an integer between 0 and 2. Current value: {}'.format(verbose)

        # 1. potentially scale input
        X_test = X.copy()
        if self.scale_X:
            X_test = self.transform_X_scaler(X_test)

        # 2. perform actual prediction 
        prediction = self.model.predict(X_test, verbose=verbose)

        # 3. Potentially reverse transform output
        if self.scale_y:
            prediction = self.inverse_transform_y_scaler(prediction)

        return prediction

    def load_keras_model(self, folder):
        '''
            Load keras model from folder and overwrite existing instance.

            Args:
                folder (str): Path to folder containing model
        '''
        assert os.path.exists(folder), 'Model.set_keras_model(): Folder does not exist. Current value: {}'.format(folder)

        # load keras model object and set it to the model instance
        try: 
            keras_model = keras.models.load_model(folder, compile=True)
            self.model = keras_model

            # restore dimensions of the network
            self.input_dim = self.model.input_shape[1]
            self.output_dim = self.model.output_shape[1]
            self.model_initialized = True

        except Exception as e:
            self.__log__('Model.load_keras_model(): Unable to load model from folder. Error: {}'.format(e), level='error')
    
    def load_X_scaler(self, filepath, overwrite_scaler_configuration:bool=True):
        ''' 
            Load y_scaler from folder and set it to the model instance.
            NOTE: it is not checked whether the dimensions of the scaler match the input dimensions of the model.

            Args:
                filepath (str): Path to file containing X_scaler
                overwrite_scaler_configuration: boolean to enfore usage of X_scaler after loading it
                    - the model can be initialized with scale_x=False
                    - but when the scaler is loaded we assume that you want to apply it from now on
                    - when you do not want this behavior, set overwrite_scaler_configuration to False

        '''
        assert file_exists(filepath), 'Model.load_X_scaler(): Filepath does not exist: {}'.format(filepath)
        try: 
            scaler = pickle.load(open(filepath, 'rb'))
            self.Xscaler = scaler
        except Exception as e:
            self.__log__('Model.load_X_scaler(): Unable to load scaler. Error: {}'.format(e), level='error')
        if overwrite_scaler_configuration: 
            self.scale_X = True

    def load_y_scaler(self, filepath, overwrite_scaler_configuration:bool=True):
        ''' 
            Load y_scaler from folder and set it to the model instance.
            NOTE: it is not checked whether the dimensions of the scaler match the input dimensions of the model.

            Args:
                filepath (str): Path to file containing y_scaler
                overwrite_scaler_configuration: boolean to enfore usage of y_scaler after loading it
                    - the model can be initialized with scale_y=False
                    - but when the scaler is loaded we assume that you want to apply it from now on
                    - when you do not want this behavior, set overwrite_scaler_configuration to False

        '''
        assert file_exists(filepath), 'Model.load_y_scaler(): Filepath does not exist: {}'.format(filepath)
        try: 
            scaler = pickle.load(open(filepath, 'rb'))
            self.yscaler = scaler
        except Exception as e:
            self.__log__('Model.load_y_scaler(): Unable to load scaler. Error: {}'.format(e), level='error')
        if overwrite_scaler_configuration: 
            self.scale_y = True
            
    def save_x_scaler(self):
        '''
            Saves the X_scaler object.
        '''
        
        if self.Xscaler is not None:
            path = self.handler.path('scaler') + 'Xscaler.sav'
            pickle.dump(self.Xscaler, open(path, 'wb'))
            self.__log__('Model.save_x_scaler(): X_Scaler saved as: {}'.format(path), level = 'info')
        else: 
            self.__log__('Model.save_x_scaler(): X_Scaler does not exists and therefore cannot be saved.', level='warning')

    def save_y_scaler(self):
        '''
            Saves the y_scaler object.
        '''
        
        if self.Yscaler is not None:
            path = self.handler.path('scaler') + 'Yscaler.sav'
            pickle.dump(self.Yscaler, open(path, 'wb'))
            self.__log__('Model.save_y_scaler(): Y_Scaler saved as: {}'.format(path), level = 'info')
        else: 
            self.__log__('Model.save_y_scaler(): Y_Scaler does not exists and therefore cannot be saved.', level='warning')
    
    def get_batch_size(self):
        '''
            Return batch size of the model.
        '''
        return self.batch_size
    
    def set_batch_size(self, batch_size:int):
        ''' 
            Set batch size of the model.

            Args:
                batch_size (int): New batch size
        '''
        assert isinstance(batch_size, int) and batch_size > 0, 'Model.set_batch_size(): batch_size has to be an integer greater than 0. Current value: {}'.format(batch_size)
        self.batch_size = batch_size
        self.__log__('Model.set_batch_size(): Batch size set to {}.'.format(batch_size), level='debug')

    def get_epochs(self):
        '''
            Return number of epochs of the model.
        '''
        return self.epochs

    def set_epochs(self, epochs:int):
        '''
            Set number of epochs of the model.

            Args:
                epochs (int): New number of epochs
        '''
        assert isinstance(epochs, int) and epochs > 0, 'Model.set_epochs(): epochs has to be an integer greater than 0. Current value: {}'.format(epochs)
        self.epochs = epochs
        self.__log__('Model.set_epochs(): Number of epochs set to {}.'.format(epochs), level='debug')
            
    def get_handler(self): 
        '''
            Return handler object of the model.
        '''
        return self.handler
    
    def get_tensorboard_path(self):
        """
            Return path to current tensorboard

            Returns:
                str: path
        """
        return self.tensorboard_path

    def fit_transform_X_scaler(self, X:np.ndarray):
        ''' 
            Fit input scaler and transform data.

            Args:
                X (numpy.ndarray): X_train
            Returns:
                X_transformed (numpy.ndarray): Scaled input data
        ''' 
        assert isinstance(X, np.ndarray), 'Model.fit_transform_X_scaler(): X has to be a numpy.ndarray. Current type: {}'.format(type(X))
        scaler = StandardScaler(with_mean=False)
        X_transformed = scaler.fit_transform(X)
        self.Xscaler = scaler

        return X_transformed

    def transform_X_scaler(self, X:np.ndarray):
        """
            Scale input data with defined scaler.

            Args:
                X (numpy.ndarray): X_train or X_test

            Returns:
                X_transformed (numpy.ndarray): Scaled input data
        """
        assert isinstance(X, np.ndarray), 'Model.transform_X_scaler(): X has to be a numpy.ndarray. Current type: {}'.format(type(X))
        assert self.Xscaler is not None, 'Model.transform_X_scaler(): Xscaler is not defined. Please use fit_transform_X_scaler() first.'
        X_transformed = self.Xscaler.transform(X)

        return X_transformed
    
    def fit_transform_y_scaler(self, y:np.ndarray):
        ''' 
            Fit output scaler and transform data.

            Args:
                y (numpy.ndarray): y_train
            Returns:
                y_transformed (numpy.ndarray): Scaled output data
        ''' 
        assert isinstance(y, np.ndarray), 'Model.fit_transform_y_scaler(): y has to be a numpy.ndarray. Current type: {}'.format(type(y))
        scaler = StandardScaler(with_mean=False)
        y_transformed = scaler.fit_transform(y)

        self.Yscaler = scaler

        return y_transformed

    def transform_y_scaler(self, y:np.ndarray):
        ''' 
            Scale output data with defined scaler.

            Args:
                y (numpy.ndarray): y_train or y_test
            Returns:
                y_transformed (numpy.ndarray): Scaled ouput data
        ''' 
        assert isinstance(y, np.ndarray), 'Model.transform_y_scaler(): y has to be a numpy.ndarray. Current type: {}'.format(type(y))
        assert self.Yscaler is not None, 'Model.transform_y_scaler(): Yscaler is not defined. Please use fit_transform_y_scaler() first.'
        y_transformed = self.Yscaler.transform(y)

        return y_transformed
    
    def inverse_transform_y_scaler(self, y_transformed:np.ndarray):
        ''' 
            Inverse scale output if output scaler is used.

            Args:
                y_transformed (array): Scaled ouput

            Returns:
                y (array): Unscaled output
        ''' 
        assert isinstance(y_transformed, np.ndarray), 'Model.inverse_transform_y_scaler(): y_transformed has to be a numpy.ndarray. Current type: {}'.format(type(y_transformed))
        assert self.Yscaler is not None, 'Model.inverse_transform_y_scaler(): Yscaler is not defined. Please use fit_transform_y_scaler() first.'
        y = self.Yscaler.inverse_transform(y_transformed)

        return y

    def Relu(self, x):
        ''' 
            Custom ReLu function. Sets negative output to zero. Only used for prediction, not for training.

            Args:
                x (Array): Array from output layer of ML model
            Returns:
                Array: Array, negative numbers are mapped to zero.
        ''' 
        return np.maximum(0,x)

    def __log__(self, msg, level='debug'): 
        ''' 
            Does the logging of a given msg. If handler object was assigned to be None, just print out is created - else the handler object is used for logging.
            Args: 
                msg: string to be logged
                level: logging level (ignored if handler is None)
        ''' 
        if self.suppress_logs: 
            return
        if self.handler is None: 
            print(str(msg))
        else: 
            self.handler.log(level, msg)
    
    def __repr__(self): 
        '''
            Overwrite string representation of Model object.
        '''
        s = '-------------------------\n'
        s += 'Class: Model\n'
        s += '-------------------------\n'
        s += 'Model Name: {}\n'.format(self.model_name)
        s += 'Fold: {}\n'.format(self.fold)
        s += 'Basepath: {}\n'.format(self.handler.basepath)
        s += 'Tensorboard Path: {}\n'.format(self.tensorboard_path)
        s += 'Supress Print-Outs: {}\n'.format(self.suppress_logs)
        s += 'Layer Shapes: {}\n'.format(self.layer_shapes)
        s += '-------------------------\n'
        s += 'Training Configurations:\n'
        s += '-------------------------\n'
        s += 'Epochs: {}\n'.format(self.epochs)
        s += 'Batch Size: {}\n'.format(self.batch_size)
        s += 'Learning Rate Schedule: {}\n'.format(self.lr_schedule)
        s += 'Initial Learning Rate: {}\n'.format(self.lr_init)
        s += 'Beta 1: {}\n'.format(self.beta_1)
        s += 'L2 Regularization Weight: {}\n'.format(self.regularizer)
        s += 'Dropout: {}\n'.format(self.dropout)
        s += 'Early Stopping Patience: {}\n'.format(self.early_stopping_patience)
        s += 'Apple Silicon: {}\n'.format(self.apple_silicon)
        s += 'Scale Input: {}\n'.format(self.scale_X)
        s += 'Scaler Output: {}\n'.format(self.scale_y)
        s += '-------------------------\n'
        # s += 'Keras Model:\n'
        # s += '-------------------------\n'
        # s += '{}\n'.format(self.model.summary())
        # s += '-------------------------\n'

        return s