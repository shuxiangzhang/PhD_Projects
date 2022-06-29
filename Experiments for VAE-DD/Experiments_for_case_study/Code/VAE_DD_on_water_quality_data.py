#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC
from scipy.special import comb as comb
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
import json
import joblib
import os,shutil
import bisect


# In[ ]:


# Read data
data = pd.read_csv('real_world_data.csv',index_col=0)
data.replace(0,np.nan,inplace=True)
data.interpolate(inplace=True)
data.index = np.arange(data.shape[0])+1
stream_train = data.iloc[:300,:]
data_stream = data.reset_index(drop=True)
data_stream.index = np.arange(data_stream.shape[0])+1



# Define method
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def Vae(data_dim,params):    
    # Build the encoder
    n_layers = len(params)-1
    encoder_inputs = keras.Input(shape=(data_dim,))
    x = encoder_inputs
    for i in range(n_layers-1):
        x = layers.Dense(params['x'+str(i+1)],activation="relu")(x)

    z_mean = layers.Dense(params['x'+str(n_layers)],name="z_mean")(x)
    z_log_var = layers.Dense(z_mean.shape[1],name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    # encoder.summary()


    # Build the decoder
    latent_inputs = keras.Input(shape=(z_mean.shape[1],))
    y = latent_inputs
    for i in range(n_layers-1):
        y = layers.Dense(params['x'+str(n_layers-i-1)],activation="relu")(y)

    decoder_outputs = layers.Dense(data_dim,activation="sigmoid")(y)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")   


    vae_outputs = decoder(encoder(encoder_inputs)[2])
    vae = Model(encoder_inputs, vae_outputs, name='vae')


    reconstruction_loss = keras.losses.mean_squared_error(encoder_inputs, vae_outputs)
    reconstruction_loss *= data_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    return encoder, decoder, vae

class VAE_DD(ABC):

    def __init__(self, delta, num_epochs,training_size, batch_size, win_size, data_dim):
        self.num_epochs = num_epochs
        self.training_size = training_size
        self.win_size = win_size
        self.batch_size = batch_size
        self.detection_window = []
        self.training_window = []
        self.drift_status = False
        self.delta = delta
        self.hoeffding_bound = None
        self.drift_threshold = None
        self.drifts = []
        self.counter = 0
        self.model = None
        self.scaler = None
        self.instance_counter = 0
        self.exception_threshold = None
        self.p_train = None
        self.d = data_dim
        self.anomaly_rate = 0
        self.train_preds = None
        self.detection_preds = None
        self.median = None
        self.med_abs_deviation = None

    @staticmethod
    def get_predictions(model_name, x_test_scaled, threshold):
        z_mean, z_log_var, z = model_name[0].predict(x_test_scaled)

        predictions = model_name[1](z)
        # provides losses of individual instances
        errors = np.mean((x_test_scaled-predictions)**2,axis=1)
        # 0 = anomaly, 1 = normal

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
       
        z_mean, _, z = model_name[0].predict(x_train_scaled)
        reconstructions = model_name[1](z)
        reconstruction_error = np.mean((x_train_scaled-reconstructions)**2,axis=1)
        threshold = np.quantile(reconstruction_error, 0.9)
        return threshold

    # Esitimate anomaly rate.
    @staticmethod
    def mad(points, thresh=3.5):
        """
        Returns a boolean array with True if points are outliers and False 
        otherwise.

        Parameters:
        -----------
            points : An numobservations by numdimensions array of observations
            thresh : The modified z-score to use as a threshold. Observations with
                a modified z-score (based on the median absolute deviation) greater
                than this value will be classified as exceptions.

        Returns:
        --------
            mask : A numobservations-length boolean array.

        References:
        ----------
            Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
            Handle Outliers", The ASQC Basic References in Quality Control:
            Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
        """
        if len(points.shape) == 1:
            points = points[:,None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh, median, med_abs_deviation
    @staticmethod
    def exception_result(error, median, med_abs_deviation, thresh=3.5):
        diff = np.abs(error-median)
        modified_z_score = 0.6745 * diff / med_abs_deviation
        return modified_z_score <= thresh
    
    def training_autoencoder(self, train_data, d, params):
        train_data = pd.DataFrame(train_data).reset_index(drop=True)
        scaler_1 = MinMaxScaler(feature_range=(0, 1))
        x_train_scaled_1 = scaler_1.fit_transform(train_data)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        model_1 = Vae(d,params)
        model_1[2].compile(
            optimizer=keras.optimizers.Adam(learning_rate = params['lr']),
            loss='mean_squared_error',
            metrics=['mean_squared_error'])
        model_1[2].fit(x_train_scaled_1,x_train_scaled_1,epochs=self.num_epochs, batch_size=self.batch_size, verbose=False,callbacks=[callback], validation_split=0.2)
        threshold_1 = self.find_threshold(model_1, x_train_scaled_1)
        preds_1,reconstruction_error_1 = self.get_predictions(model_1, x_train_scaled_1, threshold_1) 
        anomaly_index = np.where(self.mad(reconstruction_error_1)[0])[0]
        anomaly_rate = anomaly_index.size/self.training_size
        clean_data = train_data.drop(anomaly_index)
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train_scaled = scaler.fit_transform(clean_data)
        model = Vae(d,params)
        model[2].compile(
            optimizer=keras.optimizers.Adam(learning_rate = params['lr']),
            loss='mean_squared_error',
            metrics=['mean_squared_error'])
        model[2].fit(x_train_scaled,x_train_scaled, epochs=self.num_epochs, batch_size=batch_size, verbose=False,callbacks=[callback],validation_split=0.2)
        self.counter += 1
        threshold = self.find_threshold(model, x_train_scaled)
        preds,reconstruction_error_clean = self.get_predictions(model, x_train_scaled, threshold)
        exception = np.where(self.mad(reconstruction_error_clean)[0])[0].size
        self.model = model
        self.scaler = scaler
        self.exception_threshold = threshold
        self.train_preds = preds
        self.median = self.mad(reconstruction_error_clean)[1]
        self.med_abs_deviation = self.mad(reconstruction_error_clean)[2]
        p_train = exception / clean_data.shape[0]
        self.anomaly_rate = max(anomaly_rate, self.anomaly_rate)
        self.p_train = p_train
        return (anomaly_index, reconstruction_error_1, threshold_1, x_train_scaled,threshold,reconstruction_error_clean)
    
    def detect(self, instance):
        self.instance_counter += 1
        data_point = self.scaler.transform(np.array(instance).reshape(1, self.d))
        pred, error = self.get_predictions(self.model, data_point, self.exception_threshold)
        pred = self.exception_result(error, self.median, self.med_abs_deviation, thresh=3.5)
        if self.drift_status:
            if len(self.training_window) < self.training_size:
                self.training_window.append(instance)
            else:
                self.training_autoencoder(self.training_window, self.d, params)
                self.drift_status = False
                self.training_window = []
        elif len(self.detection_window) < self.win_size:
            self.detection_window.append(pred)
        else:
            p_now = 1 - np.sum(self.detection_window) / self.win_size
            self.detection_preds = self.detection_window
            m = 1/(1/len(self.train_preds)+1/len(self.detection_preds))
            self.hoeffding_bound = np.sqrt(
            1/(2*m) * np.log(4 / self.delta))
            self.drift_threshold = self.hoeffding_bound+self.anomaly_rate
            if abs(p_now - self.p_train) > self.drift_threshold:
                self.drifts.append(self.instance_counter)
                print(f'Drift was detected at index:{self.instance_counter}')
                self.drift_status = True
                self.detection_window = []
            else:
                self.detection_window.append(pred)
                self.detection_window.pop(0)
                

# Drift detection
np.random.seed(0)
import warnings
tf.random.set_seed(0)
learning_rate = 0.001
num_epochs = 500
training_size = 300
batch_size = 64
win_size = 100
delta = 0.05
dim = 5
params = {'lr': learning_rate, 'x1': int(dim*0.8), 'x2': int(dim*0.4), 'x3': int(dim*0.2)}
vae_dd = VAE_DD(delta,num_epochs,training_size, batch_size, win_size, dim)
vae_dd.training_autoencoder(stream_train,dim,params)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for i in range(data_stream.shape[0]):
        print(i)
        y = fddva.detect(data_stream.iloc[i,])
    print(fddva.drifts)

