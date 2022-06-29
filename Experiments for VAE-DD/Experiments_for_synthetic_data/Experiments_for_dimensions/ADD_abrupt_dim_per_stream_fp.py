#!/usr/bin/env python
# coding: utf-8

from abc import ABC
import numpy as np
import pandas as pd
from math import comb
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from itertools import combinations
import json

def data_generator(mu_a,mu_b,sigma_a,sigma_b,rho_a,rho_b, dim, size):
    mean = np.random.uniform(mu_a,mu_b,dim)
    sigma = np.random.uniform(sigma_a,sigma_b,dim)
    rho = np.random.uniform(rho_a,rho_b,comb(dim,2))
    cov_matrix = np.full((dim, dim), 0.0)
    counter = 0
    for i in combinations(range(dim),2):
        cov = rho[counter] * abs(sigma[i[0]]*sigma[i[1]])
        cov_matrix[i[0],i[1]] = cov
        cov_matrix[i[1],i[0]] = cov
        counter+=1
    np.fill_diagonal(cov_matrix, sigma ** 2)
    stream = np.random.multivariate_normal(mean, cov_matrix, size,'ignore')
    return (stream, mean, sigma, rho)

def add_drifts(orig_mean,orig_sigma,orig_rho,drift_mag_mean,drift_mag_sigma,drift_mag_rho,size):
    mean = orig_mean+drift_mag_mean
    sigma = orig_sigma+drift_mag_sigma
    rho = orig_rho+drift_mag_rho
    dim = mean.size
    cov_matrix = np.full((dim, dim), 0.0)
    counter = 0
    for i in combinations(range(dim),2):
        cov = rho[counter] * abs(sigma[i[0]]*sigma[i[1]])
        cov_matrix[i[0],i[1]] = cov
        cov_matrix[i[1],i[0]] = cov
        counter+=1
    np.fill_diagonal(cov_matrix, sigma ** 2)
    stream = np.random.multivariate_normal(mean, cov_matrix, size,'ignore')
    return (stream, mean, sigma, rho)

def sigmoid(x,transition):
  return 1 / (1 + np.exp(-4/transition*(x-transition/2)))

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
                self.detection_window = self.training_window
#                 print(f'Retrain autoencoder and its new train_error is {self.train_error}')
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
#             print(f'current_error {self.current_error}')
#             print(f'previous_error {self.previous_error}')
#             print(f'train_error:{self.train_error}')
            if self.abrupt_drift_status:
                print(f'Abrupt drift was detected at {self.instance_counter}')
                self.drifts.append(self.instance_counter)
            self.detection_window.pop(0)
            self.detection_window.append(instance)

np.random.seed(0)
tf.random.set_seed(0)
learning_rate = 0.001
num_epochs = 500
training_size = 300
batch_size = 64
win_size = 200
first_concept = 9000
second_concept = 1000
mu_a,mu_b, sigma_a, sigma_b, rho_a, rho_b = (0.0,50.0,0,0.2,-1,1)
mu_c = 50.0
mu_d = 60.0
noise_degree = 0.0
drift_mag = 0
anomaly_rate = 0
transition = 0
DRIFT = {}

for dim in [10,14,18,20]:
    print(f'dim:{dim}')
    params = {'lr': learning_rate, 'x1': int(dim*0.8), 'x2': int(dim*0.4), 'x3': int(dim*0.2)}
    Drift = []
  
    for run in range(100):
        print(f'run:{run}')
        stream_0_params = data_generator(mu_a, mu_b, sigma_a, sigma_b, rho_a, rho_b, dim, (first_concept+training_size))
        stream_train = pd.DataFrame(stream_0_params[0][:training_size,:])
        stream = stream_0_params[0][training_size:,:]
        mean_0,sigma_0,rho_0,drift_mag_sigma,drift_mag_rho = stream_0_params[1],stream_0_params[2],stream_0_params[3],0,0
#         if transition != 0:
#             for i in range(transition):
#                 p = np.random.rand(1)
#                 prob = sigmoid(i,transition)
#                 if p<=prob:
#                     x = add_drifts(mean_0,sigma_0,rho_0,drift_mag,drift_mag_sigma,drift_mag_rho,1)[0]
#                     stream = np.concatenate([stream, x], axis=0)
#                 else:
#                     x = add_drifts(mean_0,sigma_0,rho_0,0,drift_mag_sigma,drift_mag_rho,1)[0]
#                     stream = np.concatenate([stream, x], axis=0)         
        stream_next_params = add_drifts(mean_0,sigma_0,rho_0,drift_mag,drift_mag_sigma,drift_mag_rho,second_concept)
        stream_next = stream_next_params[0]
        stream = np.concatenate([stream, stream_next], axis=0)
        stream = pd.DataFrame(stream)
        
#         # Add noise to the data.
#         if noise_degree != 0:
#             for data_dim in range(dim):
#                 for i in range(training_size):
#                     stream_train.iloc[i,data_dim] = stream_train.iloc[i,data_dim]+np.random.normal(loc=0, scale=noise_degree*drift_mag,size=1)[0]                
#                 for i in range(stream.shape[0]):
#                             stream.iloc[i,data_dim] = stream.iloc[i,data_dim]+np.random.normal(loc=0, scale=noise_degree*drift_mag,size=1)[0]   
       
#         # Add anomalies to the data.
#         if anomaly_rate != 0:
#             anomaly_index_1 = np.random.choice(list(range(training_size)), int(anomaly_rate * training_size), replace=False)
#             for i in anomaly_index_1:
#                 stream_train.iloc[i,] = data_generator(mu_c,mu_d,sigma_a,sigma_b,rho_a,rho_b,dim,1)[0]            
#             anomaly_index_2 = np.random.choice(list(range(stream.shape[0])), int(anomaly_rate * stream.shape[0]), replace=False)
#             for i in anomaly_index_2:
#                 stream.iloc[i,] = data_generator(mu_c,mu_d,sigma_a,sigma_b,rho_a,rho_b,dim,1)[0]
    
        add_abrupt = ADD_ABRUPT(num_epochs, batch_size, win_size, dim)
        add_abrupt.training_autoencoder(stream_train,dim, params)
        memory = []
        for i in range(stream.shape[0]):
            add_abrupt.detect(stream.iloc[i,])

        Drift.append(add_abrupt.drifts)
    DRIFT[dim] = Drift

    file_name_1 = 'add_abrupt_dim_'+str(dim)
    a_file = open(file_name_1, "w")
    json.dump(Drift, a_file)
    a_file.close()

file_name_2 = 'add_abrupt_dim_total_'+str(dim)
a_file = open(file_name_2, "w")
json.dump(DRIFT, a_file)
a_file.close()




