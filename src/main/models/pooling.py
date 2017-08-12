#*****************************
# This file implements the 
# pooling structure between
# the two layers of LSTM,
# aggregating the individual
# level latent output into
# group level input
#
# Si Peng
# sipeng@adobe.com
# Jul. 21 2017
#*****************************


##### import modules ###########################
import pandas as pd
import numpy as np
import h5py
import os

from utils import *
###############################################


##### load data ###############################
cwd = os.getcwd()
input_path = cwd + '/data_input/'

DATA_PATH = '/largedatadrive/dsnp/sipeng/data/sample/oneyear/'
#DATA_PATH = '/Users/sipeng/Downloads/output_si/' # for load developing
train_path = DATA_PATH + 'train.hdf5'
valid_path = DATA_PATH + 'valid.hdf5'
test_path  = DATA_PATH + 'test.hdf5'

x_train_pool = np.load(input_path + 'train_pool.npy')
x_valid_pool = np.load(input_path + 'valid_pool.npy')
x_test_pool  = np.load(input_path + 'test_pool.npy')

# load group information and churn labels
train_cid, y_train = load_pooling(train_path)
valid_cid, y_valid = load_pooling(valid_path)
test_cid, y_test   = load_pooling(test_path)
###############################################



##### pool the data by group ##################
# x and y are masked by -1.0, both are float64 type
x_train_lstm2, y_train_lstm2, sw_train = pool(x_train_pool, train_cid, y_train)
x_valid_lstm2, y_valid_lstm2, sw_valid = pool(x_valid_pool, valid_cid, y_valid)
x_test_lstm2, y_test_lstm2, sw_test    = pool(x_test_pool, test_cid, y_test)

# save the data to files
np.save(input_path + 'train_lstm2.npy', x_train_lstm2)
np.save(input_path + 'valid_lstm2.npy', x_valid_lstm2)
np.save(input_path + 'test_lstm2.npy', x_test_lstm2)

np.save(input_path + 'y_train_lstm2.npy', y_train_lstm2)
np.save(input_path + 'y_valid_lstm2.npy', y_valid_lstm2)
np.save(input_path + 'y_test_lstm2.npy', y_test_lstm2)

np.save(input_path + 'sw_train_lstm2.npy', sw_train)
np.save(input_path + 'sw_valid_lstm2.npy', sw_valid)
np.save(input_path + 'sw_test_lstm2.npy', sw_test)
###############################################




