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
        self.drift_threshold = 0.0002
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
#                 print(Q_c)
                if Q_c > self.drift_threshold:
                    self.drift_status = True
                    print(f'Drift was detected at index:{self.instance_counter}')
#                     self.training_autoencoder(pd.DataFrame(self.cur_window), self.d)
#                     print(f'Retrain autoencoder and its new train_error is {self.train_error}')
                    self.drifts.append(self.instance_counter)
                    self.reset()
                    self.detection_window.append(error)


# In[4]:


## import json
np.random.seed(0)
tf.random.set_seed(0)
learning_rate = 0.001
num_epochs = 500
training_size = 300
batch_size = 64
win_size = 100
first_concept = 9000
second_concept = 1000
mu_a,mu_b, sigma_a, sigma_b, rho_a, rho_b = (0.0,50.0,0,0.2,-1,1)
mu_c = 50.0
mu_d = 60.0
transition = 0
drift_mag = 0.1
dim = 10
transition = 0
noise_degree = 0
anomaly_rate = 0
DRIFT = {}
TIME = {}
MEM = {}
for dim in [6,10,14,18,20]:
    print(f'dim:{dim}')
    params = {'lr': learning_rate, 'x1': int(dim*0.8), 'x2': int(dim*0.4), 'x3': int(dim*0.2)}
    Drift = []
    Time = []
    Mem = []
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
#                     stream.iloc[i,data_dim] = stream.iloc[i,data_dim]+np.random.normal(loc=0, scale=noise_degree*drift_mag,size=1)[0]   
       
#         # Add anomalies to the data.
#         if anomaly_rate != 0:
#             anomaly_index_1 = np.random.choice(list(range(training_size)), int(anomaly_rate * training_size), replace=False)
#             for i in anomaly_index_1:
#                 stream_train.iloc[i,] = data_generator(mu_c,mu_d,sigma_a,sigma_b,rho_a,rho_b,dim,1)[0]            
#             anomaly_index_2 = np.random.choice(list(range(stream.shape[0])), int(anomaly_rate * stream.shape[0]), replace=False)
#             for i in anomaly_index_2:
#                 stream.iloc[i,] = data_generator(mu_c,mu_d,sigma_a,sigma_b,rho_a,rho_b,dim,1)[0]
        add_trend = ADD_TREND(num_epochs, batch_size, win_size, dim)
        add_trend.training_autoencoder(stream_train,dim, params)
        memory = []
        start_time = time.time()
        for i in range(stream.shape[0]):
            add_trend.detect(stream.iloc[i,])
            memory.append(asizeof.asizeof(add_trend))
        end_time = time.time()
        t = end_time-start_time
        Time.append(t)
        Mem.append(np.mean(memory))
        Drift.append(add_trend.drifts)
    MEM[dim] = Mem
    TIME[dim] = Time
    DRIFT[dim] = Drift
        
    file_name_1 = 'add_trend_dim_'+str(dim)
    a_file = open(file_name_1, "w")
    json.dump(Drift, a_file)
    a_file.close()
    
    
    file_name_2 = 'add_trend_dim_mem_'+str(dim)
    b_file = open(file_name_2, "w")
    json.dump(Mem, b_file)
    b_file.close()
    
    file_name_3 = 'add_trend_dim_time_'+str(dim)
    c_file = open(file_name_3, "w")
    json.dump(Time, c_file)
    c_file.close()
    
file_name_1 = 'add_trend_dim_total_'+str(dim)
a_file = open(file_name_1, "w")
json.dump(DRIFT, a_file)
a_file.close()


file_name_2 = 'add_trend_dim_mem_total_'+str(dim)
b_file = open(file_name_2, "w")
json.dump(MEM, b_file)
b_file.close()

file_name_3 = 'add_trend_dim_time_total_'+str(dim)
c_file = open(file_name_3, "w")
json.dump(TIME, c_file)
c_file.close()



