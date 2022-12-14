import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import os
import pandas as pd

from tensorflow.keras import backend as K

# Reference: https://www.tensorflow.org/tutorials/structured_data/time_series

# Note 1: in __init__, shift is the lead time (forecast horizon)
# Note 2: label columns is a list of strings that denote the column name(s) of your target variable(s)
# Note 3: Leave label_width = 1

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df, val_df, test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

  def plot(self, plot_col, model=None, max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
      plt.subplot(max_n, 1, n+1)
      plt.ylabel(f'{plot_col}')
      plt.plot(self.input_indices, inputs[n, :, plot_col_index],
               label='Inputs', marker='.', zorder=-10)

      if self.label_columns:
        label_col_index = self.label_columns_indices.get(plot_col, None)
      else:
        label_col_index = plot_col_index

      if label_col_index is None:
        continue

      plt.scatter(self.label_indices, labels[n, :, label_col_index],
                  edgecolors='k', label='Labels', c='#2ca02c', s=64)
      if model is not None:
        predictions = model(inputs)
        plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                    marker='X', edgecolors='k', label='Predictions',
                    c='#ff7f0e', s=64)

      if n == 0:
        plt.legend()

    plt.xlabel('Time')

  def make_dataset(self, data, batch_size=32):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=batch_size,)

    ds = ds.map(self.split_window)

    return ds

  @property
  def train(self):
    return self.make_dataset(self.train_df)

  @property
  def val(self):
    return self.make_dataset(self.val_df)

  @property
  def test(self):
    return self.make_dataset(self.test_df)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
      # No example batch was found, so get one from the `.train` dataset
      result = next(iter(self.train))
      # And cache it for next time
      self._example = result
    return result

# Reference: https://www.tensorflow.org/tutorials/structured_data/time_series

def get_jena():

  # ****************************************
  # Download and read Jena weather data set
  # ****************************************

  zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
  csv_path, _ = os.path.splitext(zip_path)

  df = pd.read_csv(csv_path)
  # Slice [start:stop:step], starting from index 5 take every 6th record.
  df = df[5::6]

  date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

  # **************
  # Data cleaning
  # **************

  wv = df['wv (m/s)']
  bad_wv = wv == -9999.0
  wv[bad_wv] = 0.0

  max_wv = df['max. wv (m/s)']
  bad_max_wv = max_wv == -9999.0
  max_wv[bad_max_wv] = 0.0

  # The above inplace edits are reflected in the DataFrame.
  # df['wv (m/s)'].min()

  # ********************
  # Feature engineering
  # ********************

  wv = df.pop('wv (m/s)')
  max_wv = df.pop('max. wv (m/s)')

  # Convert to radians.
  wd_rad = df.pop('wd (deg)')*np.pi / 180

  # Calculate the wind x and y components.
  df['Wx'] = wv*np.cos(wd_rad)
  df['Wy'] = wv*np.sin(wd_rad)

  # Calculate the max wind x and y components.
  df['max Wx'] = max_wv*np.cos(wd_rad)
  df['max Wy'] = max_wv*np.sin(wd_rad)

  # *****
  # Time
  # *****
  
  timestamp_s = date_time.map(pd.Timestamp.timestamp)

  day = 24*60*60
  year = (365.2425)*day

  df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
  df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
  df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
  df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

  return df

# Reference: https://www.tensorflow.org/tutorials/structured_data/time_series

def compile_and_fit(model, window, max_epochs, patience=2):

  # 1. window is the instanatiated object from doing window = WindowGenerator(...)

  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(#loss=tf.keras.losses.MeanSquaredError(),
                loss = nse_loss,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=max_epochs,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

# Reference: https://www.tensorflow.org/tutorials/structured_data/time_series

def split_data(df, train_fraction, val_fraction, normalization=True):

  # 1. df should be a pandas data frame

  n = len(df)
  
  train_df = df[0:int(n*train_fraction)]
  val_df = df[int(n*train_fraction):int(n*(train_fraction+val_fraction))]
  test_df = df[int(n*(train_fraction+val_fraction)):]

  if (normalization):

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

  return train_df, val_df, test_df

# Reference: https://www.tensorflow.org/tutorials/structured_data/time_series

def split_data_2(df, train_fraction, val_fraction, label_column, normalization=True):

  # 1. df should be a pandas data frame
  # 2. This function does not scale/normalize the target variable

  n = len(df)
  
  train_df = df[0:int(n*train_fraction)]
  val_df = df[int(n*train_fraction):int(n*(train_fraction+val_fraction))]
  test_df = df[int(n*(train_fraction+val_fraction)):]

  if (normalization):

    train_x_mean = train_df.loc[:, train_df.columns != label_column].mean()
    train_x_std = train_df.loc[:, train_df.columns != label_column].std()

    train_df_copy = train_df.copy()
    val_df_copy = val_df.copy()
    test_df_copy = test_df.copy()

    train_df_copy.loc[:, train_df.columns != label_column] = (train_df.loc[:, train_df.columns != label_column] - train_x_mean) / train_x_std
    val_df_copy.loc[:, train_df.columns != label_column] = (val_df.loc[:, train_df.columns != label_column] - train_x_mean) / train_x_std
    test_df_copy.loc[:, train_df.columns != label_column] = (test_df.loc[:, train_df.columns != label_column] - train_x_mean) / train_x_std

  return train_df_copy, val_df_copy, test_df_copy

# Reference: https://github.com/gee-community/ee-tensorflow-notebooks/blob/master/streamflow_prediction_lstm/ee_streamflow_prediction_lstm.ipynb

def nse_loss(y_true, y_pred):
    """
    Custom metric function to calculate the Nash-Sutcliffe model efficientcy coefficient
    From: https://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient
    Commonly used in hydrology to evaluate a model performance (NSE > 0.7 is good)
    
    Args:
        y_true: Tensor with true values from observations/labels
        y_pred: Tensor of predicted values from model
    Returns: 
       tf.Tensor of the inverted NSE value
    """
    numer = K.sum(K.pow(y_true-y_pred,2))
    denom = K.sum(K.pow(y_true-K.mean(y_true),2)) + K.epsilon()
    nse = (1 - (numer/denom))
    return -1*nse