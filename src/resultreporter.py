import numpy as np 
from sklearn.metrics import * 
import pandas as pd

# ---------------- Class: ResultReporter ----------------

class ResultReporter:
    
    def __init__(self, task_type, confusion_matrix=None, groundtruth=None, predictions=None):
        '''
            Initialization.
            NOTE: for classification, the confusion_matrix parameter cannot be None - reason is that this is the only way to ensure that classes in the data set that do not occur in the test data set are still represented

            Args:
                task_type: string to define type of task - currently supported: classification and regression 
                confusion_matrix: a quadratic two-dimensional numpy array that is the confusion matrix; cannot be None in case of classification task
                groundtruth: numpy array containing ground truth values; cannot be None in case of regression task
                predictions: numpy array containing prediction values; cannot be None in case of regression task
        '''
        assert isinstance(task_type, str), 'ResultReporter: Parameter task_type must be a string. The following values are currently supported: classification, regression'
        assert task_type in ['classification', 'regression'], 'ResultReporter: Currently only the following values are supported for the task_type parameter: classification, regression'
        self.task_type = task_type
        
        # classification task
        self.num_classes = None 
        self.confusion_matrix = None 
        self.set_confusion_matrix(confusion_matrix) # fills self.num_classes and self.confusion_matrix

        # regression task 
        self.groundtruth = None 
        self.predictions = None 
        self.set_groundtruth_and_predictions(groundtruth, predictions) # fills self.groundtruth and self.predictions 
    
    def set_confusion_matrix(self, confusion_matrix):
        '''
            Resets the confusion matrix of the class for a classification task.
            Args:
                confusion_matrix: a quadratic numpy array that is the confusion matrix
        '''
        if self.task_type == 'classification': 
            assert confusion_matrix is not None, 'ResultReporter: Confusion Matrix is not allowed to be None for a classification task.'
            assert type(confusion_matrix).__module__ == 'numpy', 'ResultReporter: Confusion Matrix must be a numpy array.'
            assert len(confusion_matrix.shape) == 2, 'ResultReporter: Confusion Matrix can only be 2-dimensional.'
            assert confusion_matrix.shape[0] == confusion_matrix.shape[1], 'ResultReporter: Confusion Matrix must be quadratic.'
            self.confusion_matrix = confusion_matrix
            self.num_classes = self.confusion_matrix.shape[0]
        else: 
            self.confusion_matrix = None 

    def set_groundtruth_and_predictions(self, np_gt, np_pred):
        '''
            Sets the ground truth and prediction values for a regression task.
            NOTE: one of the parameters np_pred or np_gt can be None, if you just want to overwrite one of the parameters. However, only works when the None-parameter was already previously assigned. 
            NOTE: handing over one parameter as None is however, NOT RECOMMENDED, since it is prone to errors.
            Args: 
                np_gt: numpy array to define the ground truth values 
                np_pred: numpy array to define the prediction values
        ''' 
        if self.task_type == 'regression': 
            
            # checking that predictions and ground truth are not both None 
            assert np_gt is not None or np_pred is not None, 'ResultReporter: Ground Truth and Predictions cannot both be None.'
            
            # checking ground truth for valid value 
            if np_gt is None: 
                assert self.groundtruth is not None, 'ResultReporter: Ground Truth cannot both be None because it was never initialized with a not-None value.'
                np_gt = self.groundtruth
            else: 
                assert type(np_gt).__module__ == 'numpy', 'ResultReporter: Ground Truth must be a numpy array.'

            # checking predictions for valid value 
            if np_pred is None: 
                assert self.predictions is not None, 'ResultReporter: Predictions cannot both be None because it was never initialized with a not-None value.'
                np_pred = self.groundtruth
            else: 
                assert type(np_pred).__module__ == 'numpy', 'ResultReporter: Predictions must be a numpy array.'
            
            # checking correct shapes 
            assert np_gt.shape == np_pred.shape, 'ResultReporter: Ground Truth [shape: {}] and Predictions [shape: {}] must be of the same shape.'.format(np_gt.shape, np_pred.shape)
            
            # do assignments 
            self.groundtruth = np_gt
            self.predictions = np_pred
        else: 
            self.groundtruth = None 
            self.predictions = None 

    def tp(self, classNum):
        '''
            Returns true positives from the confusion matrix for a given class.
            Args:
                classNum: The index of the class you are interested in
            Returns:
                The number of true positives of the confusion matrix for classification tasks, else NAN.
        '''
        return self.confusion_matrix[classNum, classNum] if self.task_type == 'classification' and self.confusion_matrix is not None else np.nan

    def fp(self, classNum):
        '''
            Returns false positives from the confusion matrix for a given class.
            Args:
                classNum: The index of the class you are interested in.
            Returns:
                The number of false positives of the confusion matrix for classification tasks, else NAN.
        '''
        return self.predPos(classNum) - self.tp(classNum) if self.task_type == 'classification' and self.confusion_matrix is not None else np.nan

    def fn(self, classNum):
        '''
            Returns false negatives from the confusion matrix for a given class.
            Args:
                classNum: The index of the class you are interested in.
            Returns:
                The number of false negatives of the confusion matrix for classification tasks, else NAN.
        '''
        return self.truthPos(classNum) - self.tp(classNum) if self.task_type == 'classification' and self.confusion_matrix is not None else np.nan

    def tn(self, classNum):
        '''
            Returns true negatives from the confusion matrix for a given class.
            Args:
                classNum: The index of the class you are interested in.
            Returns:
                The number of true negatives of the confusion matrix for classification tasks, else NAN.
        '''
        return self.numSamples() + self.tp(classNum) - self.predPos(classNum) - self.truthPos(classNum) if self.task_type == 'classification' and self.confusion_matrix is not None else np.nan

    def predPos(self, classNum):
        '''
            Args:
                classNum: The index of the class you are interested in.
            Returns:
                The number of rows (predictions) in a confusion matrix at the classNums column position (truth value) for a classification task, else NAN.
        '''
        return np.sum(self.confusion_matrix[:, classNum]) if self.task_type == 'classification' and self.confusion_matrix is not None else np.nan

    def truthPos(self, classNum):
        '''
            Args:
                classNum: The index of the class you are interested in.
            Returns:
                The number of columns (truth values) in a confusion matrix at the classNums row position (prediction) for a classification task, else NAN.
        '''
        return np.sum(self.confusion_matrix[classNum, :]) if self.task_type == 'classification' and self.confusion_matrix is not None else np.nan

    def numSamples(self):
        '''
            Returns:
                The overall number of elements (predictions) in a confusion matrix for a classification task, else NAN.
        '''
        return np.sum(self.confusion_matrix) if self.task_type == 'classification' and self.confusion_matrix is not None else np.nan

    def accuracy(self):
        '''
            Returns:
                The accuracy of all predictions of a confusion matrix for a classification task, else NAN.
        '''
        if self.task_type == 'classification' and self.confusion_matrix is not None:
            t = np.trace(self.confusion_matrix)
            return t  / self.numSamples() 
        else: 
            return np.nan

    def precision(self, classNum):
        '''
            Returns:
                The precision of all predictions of a confusion matrix for a classification task, else NAN.
        '''
        return self.tp(classNum) / max(1, self.predPos(classNum)) if self.task_type == 'classification' and self.confusion_matrix is not None else np.nan

    def recall(self, classNum):
        '''
            Returns:
                The recall of all predictions of a confusion matrix for a classification task, else NAN.
        '''
        return self.tp(classNum) / max(1, self.truthPos(classNum)) if self.task_type == 'classification' and self.confusion_matrix is not None else np.nan

    def fScore(self, classNum, alpha=1):
        '''
            Args:
                classNum: index of the class you are interested in
                alpha: alpha value
            Returns:
                The class-specific fScore of all predictions of a confusion matrix for a classification task, else NAN.
        '''
        if self.task_type == 'classification' and self.confusion_matrix is not None:
            p = self.precision(classNum)
            r = self.recall(classNum)
            if 0 in [p, r]:
                return 0
            return (1 + alpha ** 2) * p * r / (alpha ** 2 * p + r)
        else: 
            return np.nan

    def logFScoreSum(self):
        '''
            Returns:
                The sum of classwise logFScores of all predictions of a confusion matrix for a classification task, else NAN.
        '''
        return np.sum(np.log([max(0.01, self.fScore(classNum)) for classNum in range(self.num_classes)])) if self.task_type == 'classification' and self.confusion_matrix is not None else np.nan

    def logFScoreMean(self):
        '''
            Returns:
                The mean of classwise logFScores of all predictions of a confusion matrix for a classification task, else NAN.
        '''
        return 0.5 * np.mean(np.log10([max(0.01, self.fScore(classNum)) for classNum in range(self.num_classes)])) + 1 if self.task_type == 'classification' and self.confusion_matrix is not None else np.nan

    def precisionMacro(self):
        '''
            Returns:
                The mean of classwise precisions of all predictions of a confusion matrix for a classification task, else NAN.
                (i.e. metric evaluated independently for each class and then average - hence treating all classes equally)
        '''
        return np.mean([self.precision(classNum) for classNum in range(self.num_classes)]) if self.task_type == 'classification' and self.confusion_matrix is not None else np.nan

    def precisionMicro(self):
        '''
            Returns:
                The class-wise-weighted mean of classwise precisions of all predictions of a confusion matrix for a classification task, else NAN.
                (i.e. aggregate the contributions of all classes to compute the average metric)
        '''
        return np.sum([self.tp(classNum) for classNum in range(self.num_classes)]) / np.sum([self.predPos(classNum) for classNum in range(self.num_classes)]) if self.task_type == 'classification' and self.confusion_matrix is not None else np.nan

    def recallMacro(self):
        '''
            Returns:
                The mean of classwise recalls of all predictions of a confusion matrix for a classification task, else NAN.
                (i.e. metric evaluated independently for each class and then average - hence treating all classes equally)
        '''
        return np.mean([self.recall(classNum) for classNum in range(self.num_classes)]) if self.task_type == 'classification' and self.confusion_matrix is not None else np.nan

    def recallMicro(self):
        '''
            Returns:
                The class-wise-weighted mean of classwise recalls of all predictions of a confusion matrix for a classification task, else NAN.
                (i.e. aggregate the contributions of all classes to compute the average metric)
        '''
        return np.sum([self.tp(classNum) for classNum in range(self.num_classes)]) / np.sum([self.truthPos(classNum) for classNum in range(self.num_classes)]) if self.task_type == 'classification' and self.confusion_matrix is not None else np.nan

    def fScoreMacro(self, alpha=1):
        '''
            Args:
                alpha: alpha value for fScore computation
            Returns:
                The mean of classwise fScores of all predictions of a confusion matrix for a classification task, else NAN.
                (i.e. metric evaluated independently for each class and then average - hence treating all classes equally)
        '''
        return np.mean([self.fScore(classNum, alpha) for classNum in range(self.num_classes)]) if self.task_type == 'classification' and self.confusion_matrix is not None else np.nan

    def fScoreMicro(self, alpha=1):
        '''
            Args:
                alpha: alpha value for fScore computation
            Returns:
                The class-wise-weighted mean of classwise fScores of all predictions of a confusion matrix for a classification task, else NAN.
                (i.e. aggregate the contributions of all classes to compute the average metric)
        '''
        if self.task_type == 'classification' and self.confusion_matrix is not None:
            p = self.precisionMicro()
            r = self.recallMicro()
            if p == 0 or r == 0:
                return 0
            return (1 + alpha ** 2) * p * r / (alpha ** 2 * p + r)
        else: 
            return np.nan

    def meanAbsoluteError(self): 
        '''
            Returns: 
                mean absolute error between ground truth and predictions if the task type is regression, else NAN
        '''
        return mean_absolute_error(self.groundtruth,self.predictions) if self.task_type == 'regression' and self.groundtruth is not None and self.predictions is not None else np.nan
    
    def meanSquaredError(self): 
        '''
            Returns: 
                mean squared error between ground truth and predictions if the task type is regression, else NAN
        '''
        return mean_squared_error(self.groundtruth,self.predictions) if self.task_type == 'regression' and self.groundtruth is not None and self.predictions is not None else np.nan

    
    def rootMeanSquaredError(self): 
        '''
            Returns: 
                root mean squared error between ground truth and predictions if the task type is regression, else NAN
        '''
        return np.sqrt(self.meanSquaredError()) if self.task_type == 'regression' and self.groundtruth is not None and self.predictions is not None else np.nan

    def rootMeanSquaredLogError(self): 
        '''
            Returns: 
                root mean squared log error between ground truth and predictions if the task type is regression, else NAN
        '''
        return np.log(self.rootMeanSquaredError()) if self.task_type == 'regression' and self.groundtruth is not None and self.predictions is not None else np.nan

    def R2Score(self): 
        '''
            Returns: 
                R2Score between ground truth and predictions if the task type is regression, else NAN
        '''
        return r2_score(self.groundtruth,self.predictions) if self.task_type == 'regression' and self.groundtruth is not None and self.predictions is not None else np.nan

    def maxResidualError(self): 
        '''
            Returns: 
                maximum residual error between ground truth and predictions if the task type is regression, else NAN
        '''
        return max_error(self.groundtruth,self.predictions) if self.task_type == 'regression' and self.groundtruth is not None and self.predictions is not None else np.nan

    def medianAbsoluteError(self): 
        '''
            Returns: 
                median absolute error between ground truth and predictions if the task type is regression, else NAN
        '''
        return median_absolute_error(self.groundtruth,self.predictions) if self.task_type == 'regression' and self.groundtruth is not None and self.predictions is not None else np.nan

    def getResultDict(self, only_relevant_metrics=False, parameter_evaluated=None):
        '''
            Returns a dictionary with all results and metrics. 
            Args: 
                only_relevant_metrics: boolean to set to True if only the metrics that are relevant for the assigned task type should be returned 
                    NOTE: setting this to False makes sense when wanting to write CSV files in a standardized way across different tasks
                parameter_evaluated: string for adding information about results to the dictionary, if not desired set to None
        '''

        assert isinstance(only_relevant_metrics, bool), 'ResultReporter.getResultDict(): Parameter only_relevant_metrics must be of type boolean.'
        assert isinstance(parameter_evaluated, str) or parameter_evaluated is None, 'ResultReporter.getResultDict(): Parameter parameter_evaluated must be of type string or None.'

        d = {}
        
        # general information 
        d['task_type'] = self.task_type
        if parameter_evaluated is not None: 
            d['parameter_evaluated'] = parameter_evaluated
        else: 
            d['parameter_evaluated'] = ''

        # classification metrics
        if self.task_type == 'classification' or not only_relevant_metrics:
            d['logFScoreSum'] = self.logFScoreSum()
            d['logFScoreMean'] = self.logFScoreMean()
            d['precisionMacro'] = self.precisionMacro()
            d['precisionMicro'] = self.precisionMicro()
            d['recallMacro'] = self.recallMacro()
            d['recallMicro'] = self.recallMicro()
            d['accuracy']  = self.accuracy()
            d['fScoreMacro'] = self.fScoreMacro()
            d['fScoreMicro'] = self.fScoreMicro()

        # regression metrics 
        if self.task_type == 'regression' or not only_relevant_metrics:
            d['meanAbsoluteError'] = self.meanAbsoluteError()
            d['meanSquaredError'] = self.meanSquaredError()
            d['rootMeanSquaredError'] = self.rootMeanSquaredError()
            d['rootMeanSquaredLogError'] = self.rootMeanSquaredLogError()
            d['R2Score'] = self.R2Score()
            d['maxResidualError'] = self.maxResidualError()
            d['medianAbsoluteError'] = self.medianAbsoluteError()
        return d

    def getResultDataFrame(self, only_relevant_metrics=False, parameter_evaluated=None):
        '''
            Returns a dataframe with all results and metrics. 
            Args: 
                only_relevant_metrics: boolean to set to True if only the metrics that are relevant for the assigned task type should be returned 
                    NOTE: setting this to False makes sense when wanting to write CSV files in a standardized way across different tasks
                parameter_evaluated: string for adding information about results to the dictionary, if not desired set to None
        '''

        assert isinstance(only_relevant_metrics, bool), 'ResultReporter.getResultDict(): Parameter only_relevant_metrics must be of type boolean.'
        assert isinstance(parameter_evaluated, str) or parameter_evaluated is None, 'ResultReporter.getResultDict(): Parameter parameter_evaluated must be of type string or None.'
        df = pd.DataFrame()
        for k,v in self.getResultDict(only_relevant_metrics=only_relevant_metrics, parameter_evaluated=parameter_evaluated).items(): 
            df[k] = pd.Series(v)
        return df

    def get_relevant_metrics(self): 
        '''
            Returns a list of strings with relevant metrics for the assigned task type. 
        '''
        d = {
            'classification': ['logFScoreSum', 'logFScoreMean', 'precisionMacro', 'precisionMicro', 'recallMacro', 'recallMicro', 'accuracy', 'fScoreMacro', 'fScoreMicro'],
            'regression' : ['meanAbsoluteError', 'medianAbsoluteError', 'meanSquaredError', 'rootMeanSquaredError', 'rootMeanSquaredLogError', 'R2Score', 'maxResidualError']
        } 
        return d[self.task_type]