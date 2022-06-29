#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from timeit import default_timer as timer
import os
from skimage.io import imread
from skimage.metrics import mean_squared_error
import pandas as pd
import numpy as np
from random import seed, shuffle
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

# Define method
def IBDD(train_X, test_X, window_length, consecutive_values):

    files2del = ['w1.jpeg', 'w2.jpeg', 'w1_cv.jpeg', 'w2_cv.jpeg']
    print(test_X.shape)
    n_runs = 20
    superior_threshold, inferior_threshold, nrmse = find_initial_threshold(train_X, window_length, n_runs)
    threshold_diffs = [superior_threshold - inferior_threshold]

    recent_data_X = train_X.iloc[-window_length:].copy()

    drift_points = []
    w1 = get_imgdistribution("w1.jpeg", recent_data_X)
    lastupdate = 0
    start = timer()
    print('IBDD Running...')
    for i in range(test_X.shape[0]): 
        recent_data_X.drop(recent_data_X.index[0], inplace=True, axis=0)
        recent_data_X = recent_data_X.append(test_X.iloc[[i]], ignore_index=True)

        w2 = get_imgdistribution("w2.jpeg", recent_data_X)

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
    w1_cv = get_imgdistribution("w1_cv.jpeg", w1)

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
        w2_cv = get_imgdistribution("w2_cv.jpeg", w2)
        

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

# Read data
data = pd.read_csv('real_world_data.csv',index_col=0)
data.replace(0,np.nan,inplace=True)
data.interpolate(inplace=True)
data.index = np.arange(data.shape[0])+1
stream_train = data.iloc[:300,:]
data_stream = data.iloc[300:,:].reset_index(drop=True)
data_stream.index = np.arange(data_stream.shape[0])+1


# Drift detection
noise_degree = 0
training_size = 300
window_size = 300 #window parameter to build the images for comparison
epsilon = 3 #number of MSD values above/below threshold
dim = 5
drifts = IBDD(stream_train,data_stream,window_size, epsilon)

