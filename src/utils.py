import os
import numpy as np
import pandas as pd
from datetime import datetime

def mkdir(path):
    '''
        Checks for existence of a path and creates it if necessary: 
        Args: 
            path: (string) path to folder
    '''
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError as e:
            if not os.path.exists(path):
                raise

def file_exists(fp):
    '''
        Checks if a file exists. 
        Args: 
            fp: path to file 
        Returns: 
            True if file exists, otherwise False 
    '''
    return os.path.exists(fp)


def folder_exists(fp):
    '''
        Checks if a folder exists. 
        Args: 
            fp: path to file 
        Returns: 
            True if folder exists, otherwise False 
    '''
    return os.path.isdir(fp)


def create_heatmap_data(df:pd.DataFrame, column:str):
    '''
        Creates a heatmap data from a dataframe with a datetime index. 
        NOTE: This method assumes that the dataframe has a datetime index and that the column name is valid.
        Args: 
            df: (dataframe) dataframe with datetime index
            column: (string) column name with energy values 
        Returns:
            data: (dataframe) dataframe with time as index and date as columns
    '''
    assert column in df.columns.values, f"create_heatmap_data(): Column {column} not in dataframe"
    data_df = df.copy()
    data_df["date"] = df.index.date
    data_df["time"] = df.index.time
    # Using only the first datapoint instead of summing avoids unrealistically height demand 
    get_first = lambda x: x.iloc[0] 
    # Pivot dates and times to create a two dimensional representation
    data = data_df.pivot_table(index='time', columns='date', values=column, aggfunc=get_first, dropna=False) 
    return data


def create_sliding_windows_from_data_series(data:np.array, window_size:int, window_overlap:int=0, reshape:bool=False):
    '''
        Helper funtion which slices data series into chunks by sliding windows.

        Args: 
            data (numpy.ndarray): data to be sliced in chunks
            window_size (int): length of the chunk / window
            window_overlap (int): length of overlap - default is no overlap
            reshape (boolean): whether to reshape the windowed data to 2D per window or keep it 1D per window - default: False
                - if False, shape will be (num_windows, window_size*data.shape[1] i.e. columns*window_size)
                - if True, shape will be (num_windows, window_size, data.shape[1] i.e. columns)
        Returns: 
            windowed data (numpy.ndarray)

    '''
    assert data.ndim == 1 or data.ndim == 2
    if data.ndim == 1:
        data = data.reshape((-1,1))

    # get the number of overlapping windows that fit into the data
    num_windows = (data.shape[0] - window_size) // (window_size - window_overlap) + 1
    overhang = data.shape[0] - (num_windows*window_size - (num_windows-1)*window_overlap)

    # if there's overhang, need an extra window and a zero pad on the data
    if overhang != 0:
        num_windows += 1
        newdata = np.zeros((num_windows*window_size - (num_windows-1)*window_overlap,data.shape[1]))
        newdata[:data.shape[0]] = data
        data = newdata

    sz = data.dtype.itemsize
    ret = np.lib.stride_tricks.as_strided(data, shape=(num_windows,window_size*data.shape[1]), strides=((window_size-window_overlap)*data.shape[1]*sz,sz))
    if reshape:
        return ret.reshape((num_windows,-1,data.shape[1]))
    else: 
        return ret


def create_data_series_from_sliding_windows(chunk_data:np.array, window_overlap:int, df_original:pd.DataFrame=None, new_column_name:str='reversed'):
    '''

        Helper funtion which uses chunks of sliding windows with given overlap to transform it back into a data series.
        Can be seen as the reverse of create_sliding_windows_from_data_series().

        NOTE: This function is used to reverse the sliding window transformation by averaging values with overlap in sliding windows.
        NOTE: you are responsible for passing an appropriate chunked array without padded zeros in the last row.
        NOTE: recall that create_sliding_windows_from_data_series() may cut off some data at the very end of the original time series (therefore, the returned series here might be shorter than the original data) - or data frame filled with NANs at the end
        NOTE: also take care of the fact that create_features_array_from_original_data_frame() may create an array with more features - take care only provide the energy sequence related features at this point 
        Args: 
            chunk_data (np.array): numpy array with the chunked data 
            window_overlap (int): length of overlap
            df_original (pd.DataFrame): original data frame to which the reversed data series should be added as a new column (OPTIONAL, otherwise numpy array with series will be returned)
            new_column_name (string): name of the new column in the original data frame (default is 'reversed')
        Returns: 
            either:
                data (numpy array): timeseries without overlap if no original data frame is provided
            or:
                df (pd.DataFrame): timeseries without overlap if original data frame is provided (copy of original data frame with additional column)
    '''

    for index in range(1, chunk_data.shape[0]):
        overlap_array = np.mean([chunk_data[index, 0:window_overlap], chunk_data[index-1, chunk_data.shape[1]-window_overlap::]], axis=0)
        chunk_data[index, 0:window_overlap] = overlap_array

    remaining_data = chunk_data[-1,chunk_data.shape[1]-window_overlap:]    
    chunk_data = chunk_data[:,0:chunk_data.shape[1]-window_overlap]
    data = chunk_data.reshape(1,chunk_data.shape[0]*chunk_data.shape[1])
    data = np.append(data, remaining_data)
    data = data.flatten()

    # incase the original data frame is provided, add the reversed data to it and return the data frame instead of the array 
    if isinstance(df_original, pd.DataFrame):
        assert df_original.shape[0] >= data.shape[0], 'create_data_series_from_sliding_windows(): Original data frame is shorter than the reversed data series.'
        assert isinstance(new_column_name, str), 'create_data_series_from_sliding_windows(): Parameter new_column_name must be a string.'

        df = df_original.copy().iloc[0:data.shape[0]]
        df[new_column_name] = data
        if len(df) < len(df_original): 
            df = pd.concat([df, df_original.iloc[len(df):]], axis=0)
        return df
    
    return data


def encode_time_as_2D_cyclic_feature(time, scale):
        '''
            Maps time onto 2D plane, i.e. encodes it as cyclic features to better reflect periodicity.
            Args: 
                time (int): time or date to be encoded
                scale (int): for time of day: 24*60, for day of year: 365, for day of week: 7
            Returns:
                numpy array with two columns, first colum = x values and second column = y values
        '''
        x = np.sin(2 * np.pi*time/scale)
        y = np.cos(2 * np.pi*time/scale)
        return np.array([x,y])


def create_features_array_from_original_data_frame(df:pd.DataFrame, window_length:int, window_overlap:int, key_consumption:str, dhw_value:bool=None, key_timestamp:str='timestamp', feat_energymean:bool=True, feat_time:bool=True, feat_weather:bool=True, feat_dhw:bool=True, weather_columns=['daily_avgtemp', 'hourly_avgtemp', 'daily_maxtemp', 'daily_mintemp'], remove_last_window:bool=True): 
    '''
        NOTE: this methods assumes that the weather data and smart meter data are already in one data frame together with the timestamp
        Creates numpy array with features from original data frame, i.e. sliding windows that (depending on configuration) contain: 
            - original energy sequence
            - mean of the energy sequence 
            - time of day, day of the year, and day of the week as 2D cyclic features 
            - means of weather features of whatever is provided as weather columns (default is: daily_avgtemp, hourly_avgtemp, daily_maxtemp, daily_mintemp)
            - dhw production available or not (boolean)
        Args: 
            df (pandas.DataFrame): original data frame
            window_length (int): length of the sliding window
            window_overlap (int): length of overlap
            key_consumption (string): key of the energy consumption column in the data frame
            dhw_value (bool): whether HP is responsible for domestic hot water production or not (only necessary if feat_dhw is set to True)
            key_timestamp (string): key of the timestamp column in the data frame
            feat_energymean (bool): whether to add the mean of the energy sequence as feature
            feat_time (bool): whether to add time of day, day of the year, and day of the week as features
            feat_weather (bool): whether to add weather features as features
            feat_dhw (bool): whether to add domestic hot water production as feature
            weather_columns (list): list of strings with the names of the weather columns in the data frame - NOTE: order matters! order of features will be preserved
            remove_last_window: boolean to decide if the last window should be removed because it may be zero-padded - recommended use for training, but not necessarily for testing
        Returns: 
            numpy array with features
    '''

    # create sliding window values of the energy sequence 
    assert key_consumption in df.columns, 'create_features_array_from_original_data_frame(): Key consumption {} not found in data frame.'.format(key_consumption)
    np_data = create_sliding_windows_from_data_series(df[key_consumption].values, window_length, window_overlap)

    # potentially add the energy mean as feature 
    if feat_energymean:
        np_data = np.concatenate([np_data, np.mean(np_data, axis=1).reshape(-1,1)], axis=1)

    # potentially add the time as feature
    if feat_time:
        assert(key_timestamp in df.columns), 'create_features_array_from_original_data_frame(): Timestamp column {} not found in data frame.'.format(key_timestamp)

        # get the indices that refer to the start of the windows - i.e., only process timestamp referring to the beginning of each sliding window
        num_windows = np_data.shape[0]
        num_samples = df.shape[0]
        indices = np.arange(0, num_samples, window_length-window_overlap)
        if len(indices) > num_samples:
            indices = np.delete(indices, -1)
        
        t = pd.DatetimeIndex(df[key_timestamp].iloc[indices].values)

        min_of_day = t.hour*60 + t.minute
        day_of_year = t.dayofyear
        day_of_week = t.dayofweek

        feature_time = encode_time_as_2D_cyclic_feature(min_of_day, 24*60)
        feature_date = encode_time_as_2D_cyclic_feature(day_of_year, 365)
        feature_day = encode_time_as_2D_cyclic_feature(day_of_week, 7)

        np_data = np.concatenate((np_data, feature_time[:, 0:num_windows].T), axis = 1)
        np_data = np.concatenate((np_data, feature_date[:, 0:num_windows].T), axis = 1)
        np_data = np.concatenate((np_data, feature_day[:, 0:num_windows].T), axis=1)

    # potentially add weather features 
    if feat_weather:

        assert isinstance(weather_columns, list) and len(weather_columns) > 0, 'create_features_array_from_original_data_frame(): Parameter weather_columns must be a list of strings if feat_weather is set to True.'

        # for each weather column, create sliding windows and add the mean of each window as feature
        for weather_col in weather_columns:
            assert weather_col in df.columns, 'create_features_array_from_original_data_frame(): Weather column {} not found in data frame.'.format(weather_col)
            np_weather_sliding_windows = create_sliding_windows_from_data_series(df[weather_col].values, window_length, window_overlap)
            np_data = np.concatenate((np_data, np.mean(np_weather_sliding_windows, axis=1).reshape(num_windows, 1)), axis=1)

    # potentially add dhw production feature
    if feat_dhw:
        assert dhw_value is not None and isinstance(dhw_value, bool), 'create_features_array_from_original_data_frame(): Parameter dhw_value must be set to a boolean value if feat_dhw is set to True.'
        if dhw_value: # if dhw production, add ones
            np_data = np.concatenate((np_data, np.ones((num_windows, 1))), axis=1)
        else: # if no dhw production, add zeros
            np_data = np.concatenate((np_data, np.zeros((num_windows, 1))), axis=1)

    # always remove last row because it may be incomplete and filled up with zeros 
    if remove_last_window: 
        np_data = np.delete(np_data, -1, axis=0)

    return np_data


def create_unique_model_name(layer_shapes, epochs, batch_size, learning_rate_init, beta_1, fold): 
    '''
        Creates a unique model name based on the hyperparameters and the fold. 
        Args: 
            layer_shapes: list of integers, defining the number of neurons in each layer
            epochs: integer, number of epochs
            batch_size: integer, batch size
            learning_rate_init: float, initial learning rate
            beta_1: float, beta_1 parameter of the Adam optimizer
            fold: integer, the fold number
        Returns:
            string, the unique model name
    '''
    s = str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S")) 
    s += '_layers-'+str(layer_shapes).replace(' ', '')
    s += '_epochs-'+str(epochs)
    s += '_batchsize-'+str(batch_size)
    s += '_lr-'+str(learning_rate_init)
    s += '_beta1-'+str(beta_1)
    s += '_fold'+str(fold)
    return s 