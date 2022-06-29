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


class ADD_TREND(ABC):

    def __init__(self, num_epochs, batch_size, win_size,data_dim):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.win_size = win_size
        self.training_size = 200
        self.detection_window = []
        self.training_window = []
        self.d = data_dim
        self.instance_counter = 0
        self.model = None
        self.scaler = None
        self.threshold = None
        self.drift_status = False
        self.d = data_dim
        self.drifts = []
        self.C = 0
        self.TC = 0
        self.T = 0
        self.T_2 = 0
        self.n = 0
        self.t = 0
        self.forgetting_factor = 0.998
        self.drift_threshold = 0.00002
    def reset(self):
        self.C = 0
        self.TC = 0
        self.T = 0
        self.T_2 = 0
        self.n = 0
        self.t = 0
        self.detection_window = []   
        
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
        self.model = model
        self.scaler = scaler
        self.threshold = threshold

    def detect(self, instance):
#         print(f'Index:{self.instance_counter}')
        self.instance_counter += 1
        data_point = self.scaler.transform(np.array(instance).reshape(1, self.d))
        pred, error = self.get_predictions(self.model, data_point, self.threshold)
        if self.drift_status:
            if len(self.training_window) < self.training_size:
                self.training_window.append(instance)
            else:
                self.training_autoencoder(pd.DataFrame(self.training_window), self.d, params)
                self.drift_status = False
                self.training_window = []
        elif len(self.detection_window)< self.win_size:
                self.detection_window.append(error)
        else:
            self.t = self.t+1
            self.detection_window.pop(0)
            self.detection_window.append(error)
            mean_error = np.mean(self.detection_window)
            C_m = mean_error
            self.C = self.forgetting_factor *self.C+C_m
            self.TC = self.forgetting_factor *self.TC+self.t*C_m
            self.T = self.forgetting_factor *self.T+self.t
            self.T_2 = self.forgetting_factor *self.T_2+self.t**2
            self.n = self.forgetting_factor *self.n+1
            if self.t>=2:
                Q_c = (self.n*self.TC-self.T*self.C)/(self.n*self.T_2 - self.T**2)
                if Q_c > self.drift_threshold:
                    self.drift_status = True
                    print(f'Drift was detected at index:{self.instance_counter}')
                    self.drifts.append(self.instance_counter)
                    self.reset()
                    self.detection_window.append(error)

# Data Preprocessing
data = pd.read_csv('real_world_data.csv',index_col=0)
data.replace(0,np.nan,inplace=True)
data.interpolate(inplace=True)
data.index = np.arange(data.shape[0])+1
stream_train = data.iloc[:300,:]
data_stream = data.iloc[300:,:].reset_index(drop=True)
data_stream.index = np.arange(data_stream.shape[0])+1



# Drift detection

import warnings
np.random.seed(0)
tf.random.set_seed(0)
learning_rate = 0.001
num_epochs = 500
training_size = 300
batch_size = 64
win_size = 100
dim = 5
params = {'lr': learning_rate, 'x1': int(dim*0.8), 'x2': int(dim*0.4), 'x3': int(dim*0.2)}

add_trend = ADD_TREND(num_epochs, batch_size, win_size, dim)
add_trend.training_autoencoder(stream_train,dim, params)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    for i in range(data_stream.shape[0]):
    #             print(i)
        add_trend.detect(data_stream.iloc[i,])
    print(add_trend.drifts)


# In[ ]:




