#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from abc import ABC
import numpy as np
import pandas as pd
# from math import comb
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense
from collections import OrderedDict
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from itertools import combinations
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

# 1. create a autoencoder model by subclassing Model class in tensorflow
def AutoEncoder(data_dim,params):
    """
  Parameters
  ----------
  output_units: int
    Number of output units

  code_size: int
    Number of units in bottle neck
  """
    n_layers = len(params)-1
    encoder_inputs = tf.keras.Input(shape=(data_dim,))
    x = encoder_inputs
    for i in range(n_layers-1):
        x = Dense(params['x'+str(i+1)],activation="relu")(x)
    latent_layer = Dense(params['x'+str(n_layers)],activation="relu")(x)
    # encoder.summary()

    # Build the decoder
    y = latent_layer
    for i in range(n_layers-1):
        y = Dense(params['x'+str(n_layers-i-1)],activation="relu")(y)

    decoder_outputs = Dense(data_dim,activation="sigmoid")(y)
    
    model = tf.keras.Model(inputs=encoder_inputs, outputs=decoder_outputs, name="autoencoder")
    return model


class ADD_ABRUPT(ABC):

    def __init__(self, num_epochs, batch_size, win_size,data_dim):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.win_size = win_size
        self.training_size = 300
        self.detection_window = []
        self.training_window = []
        self.train_error = None
        self.current_error = None
        self.previous_error = None
        self.instance_counter = 0
        self.abrupt_drift_status = False
        self.model = None
        self.scaler = None
        self.threshold = None
        self.abrupt_threshold = 0.005
        self.d = data_dim
        self.drifts = []
        
    @staticmethod
    def get_predictions(model_name, x_test_scaled, threshold):
        predictions = model_name.predict(x_test_scaled)
        # provides losses of individual instances
        errors = tf.keras.losses.mse(predictions, x_test_scaled)
        # 0 = anomaly, 1 = normal
        errors = errors.numpy()

        if errors.size == 1:
            anomaly_mask = errors > threshold
            if anomaly_mask:
                preds = 0.0
            else:
                preds = 1.0
        else:
            anomaly_mask = pd.Series(errors) > threshold
            preds = anomaly_mask.map(lambda x: 0.0 if x else 1.0)
        return preds, errors

    @staticmethod
    def find_threshold(model_name, x_train_scaled):
        reconstructions = model_name.predict(x_train_scaled)
        # provides losses of individual instances
        reconstruction_errors = tf.keras.losses.mse(reconstructions, x_train_scaled)

        # threshold for anomaly scores
        threshold_param = np.mean(reconstruction_errors.numpy()) + 3 * np.std(reconstruction_errors.numpy())
        return threshold_param
    
    def training_autoencoder(self, training_data, data_dim,params):
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_size = training_data.shape[0]
        x_train_scaled = scaler.fit_transform(training_data)
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['lr'])
        model = AutoEncoder(data_dim,params)
        model.compile(loss='mse', metrics=['mse'], optimizer=optimizer)
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        model.fit(
            x_train_scaled,
            x_train_scaled,
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            callbacks=[callback],
            verbose=False
        )
        threshold = self.find_threshold(model, x_train_scaled)
        _, errors = self.get_predictions(model, x_train_scaled, threshold)
        self.train_error = np.mean(errors)
        self.model = model
        self.scaler = scaler
        self.threshold = threshold

    def detect(self, instance):
        self.instance_counter += 1
        if self.abrupt_drift_status:
            if len(self.training_window) < self.training_size:
                self.training_window.append(instance)
            else:
                self.training_autoencoder(pd.DataFrame(self.training_window), self.d, params)
                self.abrupt_drift_status = False
                self.training_window = []
        elif len(self.detection_window) < self.win_size:
            self.detection_window.append(instance)
        else:
            scaled_window = self.scaler.transform(self.detection_window)
            _, errors = self.get_predictions(self.model, scaled_window, self.threshold)
            pre_errors = np.array(errors[:int(win_size/2)])
            cur_errors = np.array(errors[int(win_size/2):])          
            self.previous_error = np.mean(pre_errors)
            self.current_error = np.mean(cur_errors)
            self.abrupt_drift_status = (self.current_error - self.previous_error) > self.abrupt_threshold
            if self.abrupt_drift_status:
                print(f'Abrupt drift was detected at {self.instance_counter}')
                self.drifts.append(self.instance_counter)
                self.detection_window = []
            else:
                self.detection_window.pop(0)
                self.detection_window.append(instance)
                
                

# Data preprocessing
data = pd.read_csv('real_world_data.csv',index_col=0)
data.replace(0,np.nan,inplace=True)
data.interpolate(inplace=True)
data.index = np.arange(data.shape[0])+1
stream_train = data.iloc[:300,:]
data_stream = data.iloc[300:,:].reset_index(drop=True)


# Drift detection
np.random.seed(0)
tf.random.set_seed(0)
learning_rate = 0.001
num_epochs = 500
training_size = 300
batch_size = 64
win_size = 100
dim = 5
import warnings
params = {'lr': learning_rate, 'x1': int(dim*0.8), 'x2': int(dim*0.4), 'x3': int(dim*0.2)}
add_abrupt = ADD_ABRUPT(num_epochs, batch_size, win_size, dim)
add_abrupt.training_autoencoder(stream_train,dim, params)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for i in range(data_stream.shape[0]):
        add_abrupt.detect(data_stream.iloc[i,])
    print(add_abrupt.drifts)

