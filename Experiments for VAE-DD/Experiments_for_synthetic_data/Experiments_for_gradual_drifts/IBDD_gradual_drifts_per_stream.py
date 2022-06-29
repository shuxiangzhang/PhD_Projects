#!/usr/bin/env python
# coding: utf-8

from timeit import default_timer as timer
from skimage.io import imread
from skimage.metrics import mean_squared_error
from random import seed, shuffle
import matplotlib as mpl
import json

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 13}

mpl.rc('font', **font)

mpl.rcParams['figure.figsize']=(6,4)    #(6.0,4.0)
mpl.rcParams['font.size']=12                #10 
mpl.rcParams['savefig.dpi']=100             #72 
mpl.rcParams['figure.subplot.bottom']=.11    #.125


def IBDD(train_X, test_X, window_length, consecutive_values):

    files2del = ['wc.jpeg', 'wd.jpeg', 'wc_cv.jpeg', 'wd_cv.jpeg']
    print(test_X.shape)
    n_runs = 20
    superior_threshold, inferior_threshold, nrmse = find_initial_threshold(train_X, window_length, n_runs)
    threshold_diffs = [superior_threshold - inferior_threshold]

    recent_data_X = train_X.iloc[-window_length:].copy()

    drift_points = []
    w1 = get_imgdistribution("wc.jpeg", recent_data_X)
    lastupdate = 0
    start = timer()
    print('IBDD Running...')
    for i in range(test_X.shape[0]): 
        recent_data_X.drop(recent_data_X.index[0], inplace=True, axis=0)
        recent_data_X = recent_data_X.append(test_X.iloc[[i]], ignore_index=True)

        w2 = get_imgdistribution("wd.jpeg", recent_data_X)

        nrmse.append(mean_squared_error(w1,w2))

        if (i-lastupdate > 60):
            superior_threshold = np.mean(nrmse[-50:])+2*np.std(nrmse[-50:])
            inferior_threshold = np.mean(nrmse[-50:])-2*np.std(nrmse[-50:])
            threshold_diffs.append(superior_threshold-inferior_threshold)
            lastupdate = i

        if (all(i >= superior_threshold for i in nrmse[-consecutive_values:])):
            superior_threshold = nrmse[-1] + np.std(nrmse[-50:-1])
            inferior_threshold = nrmse[-1] - np.mean(threshold_diffs)
            threshold_diffs.append(superior_threshold-inferior_threshold)
            drift_points.append(i)
            print(f'drift was detected at {i}')
            lastupdate = i

        elif (all(i <= inferior_threshold for i in nrmse[-consecutive_values:])):
            inferior_threshold = nrmse[-1] - np.std(nrmse[-50:-1])
            superior_threshold = nrmse[-1] + np.mean(threshold_diffs) 
            threshold_diffs.append(superior_threshold-inferior_threshold) 
            drift_points.append(i)
            print(f'drift was detected at {i}')
            lastupdate = i
    return drift_points


def find_initial_threshold(X_train, window_length, n_runs):
    if window_length > len(X_train):
        window_length = len(X_train)

    w1 = X_train.iloc[-window_length:].copy()
    w1_cv = get_imgdistribution("wc_cv.jpeg", w1)

    max_index = X_train.shape[0]
    sequence = [i for i in range(max_index)]
    nrmse_cv = []
    for i in range(0,n_runs):
        # seed random number generator
        seed(i)
        # randomly shuffle the sequence
        shuffle(sequence)
        w2 = X_train.iloc[sequence[:window_length]].copy()
        w2.reset_index(drop=True, inplace=True)
        w2_cv = get_imgdistribution("wd_cv.jpeg", w2)
        

        nrmse_cv.append(mean_squared_error(w1_cv,w2_cv))
        threshold1 = np.mean(nrmse_cv)+2*np.std(nrmse_cv)
        threshold2 = np.mean(nrmse_cv)-2*np.std(nrmse_cv)
    if threshold2 < 0:
        threshold2 = 0
    return (threshold1, threshold2, nrmse_cv)



def get_imgdistribution(name_file, data):
    plt.imsave(name_file, data.transpose(), cmap = 'Greys', dpi=100)
    w = imread(name_file)
    return w


# In[2]:


from abc import ABC
import numpy as np
import pandas as pd
from math import comb
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

def data_generator(mu_0,mu_1, sigma_0,sigma_1, rho_0, rho_1, dim, size):
    mean = np.random.uniform(mu_0,mu_1,dim)
    sigma = np.random.uniform(sigma_0,sigma_1,dim)
    rho = np.random.uniform(rho_0,rho_1,comb(dim,2))
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


# In[3]:


noise_degree = 0
training_size = 300
window_size = 300 #window parameter to build the images for comparison
epsilon = 3 #number of MSD values above/below threshold
first_concept = 9000
second_concept = 1000
mu_a,mu_b, sigma_a, sigma_b, rho_a, rho_b = (0.0,50.0,0,0.2,-1,1)
mu_c = 50.0
mu_d = 60.0
anomaly_rate = 0
drift_mag = 0.1
dim = 10
DRIFT = {}
for transition in [200,400,600,800,1000]:
    print(f'transition:{transition}')
    Drift = []
    for run in range(100):
        print(f'run:{run}')
        stream_0_params = data_generator(mu_a, mu_b, sigma_a, sigma_b, rho_a, rho_b, dim, (first_concept+training_size))
        stream_train = pd.DataFrame(stream_0_params[0][:training_size,:])
        stream = stream_0_params[0][training_size:,:]
        mean_0,sigma_0,rho_0,drift_mag_sigma,drift_mag_rho = stream_0_params[1],stream_0_params[2],stream_0_params[3],0,0
        if transition != 0:
            for i in range(transition):
                p = np.random.rand(1)
                prob = sigmoid(i,transition)
                if p<=prob:
                    x = add_drifts(mean_0,sigma_0,rho_0,drift_mag,drift_mag_sigma,drift_mag_rho,1)[0]
                    stream = np.concatenate([stream, x], axis=0)
                else:
                    x = add_drifts(mean_0,sigma_0,rho_0,0,drift_mag_sigma,drift_mag_rho,1)[0]
                    stream = np.concatenate([stream, x], axis=0)         
        stream_next_params = add_drifts(mean_0,sigma_0,rho_0,drift_mag,drift_mag_sigma,drift_mag_rho,second_concept)
        stream_next = stream_next_params[0]
        stream = np.concatenate([stream, stream_next], axis=0)
        stream = pd.DataFrame(stream)
        
#         # Add noise to the data.
#         if noise_degree != 0:
#             for data_dim in range(dim):
#                 for i in range(window_size):
#                     stream_train.iloc[i,data_dim] = stream_train.iloc[i,data_dim]+np.random.normal(loc=0, scale=noise_degree*drift_mag,size=1)[0]                
#                 for i in range(stream.shape[0]):
#                             stream.iloc[i,data_dim] = stream.iloc[i,data_dim]+np.random.normal(loc=0, scale=noise_degree*drift_mag,size=1)[0]   
#         # Add anomalies to the data.
#         if anomaly_rate != 0:
#             anomaly_index_1 = np.random.choice(list(range(window_size)), int(anomaly_rate * window_size), replace=False)
#             for i in anomaly_index_1:
#                 stream_train.iloc[i,] = data_generator(mu_c,mu_d,sigma_a,sigma_b,rho_a,rho_b,dim,1)[0]            
#             anomaly_index_2 = np.random.choice(list(range(stream.shape[0])), int(anomaly_rate * stream.shape[0]), replace=False)
#             for i in anomaly_index_2:
#                 stream.iloc[i,] = data_generator(mu_c,mu_d,sigma_a,sigma_b,rho_a,rho_b,dim,1)[0]
        drifts = IBDD(stream_train,stream,window_size, epsilon)
        Drift.append(drifts)
    file_name_1 = 'ibdd_gradual_drift_'+str(transition)
    a_file = open(file_name_1, "w")
    json.dump(Drift, a_file)
    a_file.close()   
    DRIFT[transition] = Drift
file_name_1 = 'ibdd_gradual_drift_total_'+str(transition)
a_file = open(file_name_1, "w")
json.dump(DRIFT, a_file)
a_file.close()   

