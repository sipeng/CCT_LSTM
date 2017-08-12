#*****************************
# This code snippet contains
# codes to read data from
# hdf5 files, downsampling,
# and training of a
# many-to-many LSTM as well
# as evaluation
#
# Si Peng
# sipeng@adobe.com
# Jul. 20 2017
#*****************************

##### import modules ##########################
import pandas as pd
import numpy as np
import h5py
import timeit

from tabulate import tabulate
from sklearn.metrics import roc_auc_score, matthews_corrcoef, \
precision_recall_fscore_support, average_precision_score, \
roc_curve, auc, accuracy_score, precision_recall_curve

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, \
TimeDistributed, Masking
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from keras import backend as K

from utils import *
###############################################


##### load data ###############################
data_dim = 14
timesteps = 365

DATA_PATH = '/largedatadrive/dsnp/sipeng/data/sample/oneyear/'
#DATA_PATH = '/Users/sipeng/Downloads/output_si/' # for load developing
train_path = DATA_PATH + 'train.hdf5'
valid_path = DATA_PATH + 'valid.hdf5'
test_path  = DATA_PATH + 'test.hdf5'

# load training data
x_train, y_train = load_data_lstm1(train_path)
x_valid, y_valid = load_data_lstm1(valid_path)
x_test, y_test   = load_data_lstm1(test_path)

test_mask = (y_test != -1).astype(int).reshape((-1, timesteps, 1))
###############################################



##### model building **************************
model = Sequential()
model.add(Masking(mask_value=-1., input_shape=(timesteps, data_dim))) 

model.add(LSTM(20, return_sequences=True))
model.add(Dropout(0.5))

model.add(TimeDistributed(Dense(1, activation='sigmoid')))
###############################################


##### model compiling #########################
# define some paths
cwd = os.getcwd()
cache_path = cwd+'/cache/'
output_model_path = cwd+'/model_output/'

# compile the model
#adam = Adam(lr=0.01, decay=1e-02)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# create some callbacks
checkpointer = ModelCheckpoint(filepath=output_model_path+'bestmodel_lstm1.hdf5', verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
###############################################


##### model fit ###############################
print(model.summary())

History = model.fit(x_train, y_train, \
                    batch_size=100, epochs=200, shuffle=True, \
                    validation_data = (x_valid, y_valid), \
                    initial_epoch = 0, verbose=1, callbacks=[checkpointer, earlystopper])

save_history(cache_path+'history_lstm1', History)
###############################################



##### model evaluation ########################
model  = load_model(output_model_path+'bestmodel_lstm1.hdf5')
y_pred = model.predict(x_test, verbose=1, batch_size = 200)

y_pred1 = y_pred[y_test != -1]
y_test1 = y_test[y_test != -1]

# compute fpr and tpr
fpr, tpr, _ = roc_curve(y_test1, y_pred1)
auc_ROC     = auc(fpr, tpr)
pre, rec, _ = precision_recall_curve(y_test1, y_pred1)
auc_PR      = auc(rec, pre)

# plot the training and validating losses, as well as the roc curve
train, valid = load_history(cache_path+'history_lstm1.hdf5')
plot_loss_curve(output_model_path+'loss_plot_lstm1.png', train, valid)
plot_roc(output_model_path + 'roc_lstm1.png', fpr, tpr)
plot_pr(output_model_path + 'pr_lstm1.png', rec, pre)
###############################################



##### get the model output ####################
get_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-3].output])

train_output = get_layer_output([x_train, 1])[0] # a np array, 1 means training phase
valid_output = get_layer_output([x_valid, 0])[0] # 0 means test phase
test_output  = get_layer_output([x_test, 0])[0]

# squeeze the original feature from 14-dim to 2-dim: (total_session_time, total_session_count)
x_train_sq = squeeze(x_train)
x_valid_sq = squeeze(x_valid)
x_test_sq  = squeeze(x_test)

# concatenate the squeezed original feature and the latent ouput from LSTM1
x_train_pool = np.concatenate((x_train_sq, train_output), axis=2)
x_valid_pool = np.concatenate((x_valid_sq, valid_output), axis=2)
x_test_pool  = np.concatenate((x_test_sq, test_output), axis=2)

# add the mask to the data tensors
x_train_pool = mask_tensor(x_train_pool, y_train)
x_valid_pool = mask_tensor(x_valid_pool, y_valid)
x_test_pool  = mask_tensor(x_test_pool, y_test)

# save the data to files
data_output_path = cwd + '/data_input/'

np.save(data_output_path+'train_pool.npy', x_train_pool)
np.save(data_output_path+'valid_pool.npy', x_valid_pool)
np.save(data_output_path+'test_pool.npy', x_test_pool)
###############################################




"""
train_mask = (x_train != -1).mean(axis=2).reshape((-1, timesteps, 1)).astype(int)
test_mask = (y_test != -1).astype(int).reshape((-1, timesteps, 1))

uniq, cnts = np.unique(y_train, return_counts=1)

## test if the data is correct
val_mask = (x_val != -1).mean(axis=2).reshape((-1, timesteps, 1)).astype(int)
val_mask2 = (y_val != -1).astype(int).reshape((-1, timesteps, 1))

val_mask_check = (val_mask == 1)*(val_mask2 == 0)
val_mask_check = val_mask_check.reshape((-1, timesteps))
uniq, cnts = np.unique(val_mask_check, return_counts=1)
uniq  # only False, menaing the data is correct
"""