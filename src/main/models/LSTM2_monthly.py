#*****************************
# This file implements the
# LSTM2 of the hierarchical
# LSTM model, the training
# and prediction are at 
# contract level, the model
# has 12 timesteps, as contrast
# to the original 365-timestep
# model
#
# Si Peng
# sipeng@adobe.com
# Jul. 25 2017
#*****************************


##### import modules ###########################
import pandas as pd
import numpy as np
import h5py
import timeit
import tensorflow as tf
import random

random.seed(11)

from tabulate import tabulate
from sklearn.metrics import roc_auc_score, matthews_corrcoef, \
precision_recall_fscore_support, average_precision_score, \
roc_curve, auc, accuracy_score, precision_recall_curve

import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
from keras.backend.tensorflow_backend import set_session

#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
#set_session(tf.Session(config=config))

from utils import *
###############################################


##### load data ###############################
cwd = os.getcwd()
input_path = cwd + '/data_input/'

print('############### Processing Data ####################')

x_train  = np.load(input_path + 'train_lstm2_monthly.npy')
x_valid  = np.load(input_path + 'valid_lstm2_monthly.npy')
x_test   = np.load(input_path + 'test_lstm2_monthly.npy')

y_train  = np.load(input_path + 'y_train_lstm2_monthly.npy')
y_valid  = np.load(input_path + 'y_valid_lstm2_monthly.npy')
y_test   = np.load(input_path + 'y_test_lstm2_monthly.npy')

sw_train = np.load(input_path + 'sw_train_lstm2_monthly.npy')
sw_valid = np.load(input_path + 'sw_valid_lstm2_monthly.npy')
sw_test  = np.load(input_path + 'sw_test_lstm2_monthly.npy')

## scale the feautre tensor
x_train, std= scale_X(x_train)
x_valid     = scale_X(x_valid, std)
x_test      = scale_X(x_test, std)

## replace NAN with -1.0
x_train[np.isnan(x_train)] = -1.0
x_valid[np.isnan(x_valid)] = -1.0
x_test[np.isnan(x_test)]   = -1.0
###############################################



##### model building **************************

print('############### Building Model ####################')

model = Sequential()
model.add(Masking(mask_value=-1.0, input_shape=(x_train.shape[1], x_train.shape[2])))

#model.add(LSTM(40, return_sequences=True, kernel_initializer='glorot_normal'))
#model.add(Dropout(0.5))

model.add(LSTM(20, return_sequences=True, kernel_initializer='glorot_normal'))
model.add(Dropout(0.5))

#model.add(TimeDistributed(Dense(5)))
#model.add(BatchNormalization())
#model.add(Activation(activation='relu'))
#model.add(Dropout(0.5))

model.add(TimeDistributed(Dense(1, activation='sigmoid', kernel_initializer='glorot_normal')))
###############################################


##### model compiling #########################
# define some paths
cwd = os.getcwd()
cache_path = cwd+'/cache/'
output_model_path = cwd+'/model_output/'

# compile the model
adam = Adam(lr=0.002, decay=0)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'], sample_weight_mode = "temporal")

# create some callbacks
checkpointer = ModelCheckpoint(filepath=output_model_path+'bestmodel_lstm2_monthly.hdf5', verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=70, verbose=0)
###############################################


##### model fit ###############################
print(model.summary())

print('############### Fitting Model ####################')

History = model.fit(x_train, y_train, \
                    batch_size=50, epochs=300, shuffle=True, sample_weight=sw_train, \
                    validation_data = (x_valid, y_valid, sw_valid), \
                    initial_epoch = 0, verbose=1, callbacks=[checkpointer, earlystopper])

os.remove(cache_path + 'history_lstm2_monthly.hdf5')
save_history(cache_path + 'history_lstm2_monthly', History)
###############################################



##### model evaluation ########################

print('############### Evaluating Prediction ####################')

model  = load_model(output_model_path + 'bestmodel_lstm2_monthly.hdf5')
y_pred = model.predict(x_test, verbose=1, batch_size = 50)

# extract the samples with weight = 1
y_pred1 = y_pred[sw_test.reshape((sw_test.shape[0], -1, 1)) != 0.]
y_test1 = y_test[sw_test.reshape((sw_test.shape[0], -1, 1)) != 0.]

# compute fpr and tpr
fpr, tpr, _ = roc_curve(y_test1, y_pred1)
auc_ROC     = auc(fpr, tpr)
pre, rec, _ = precision_recall_curve(y_test1, y_pred1)
auc_PR      = auc(rec, pre)
print(auc_ROC, auc_PR)

# plot the training and validating losses, and ROC curve
train, valid = load_history(cache_path + 'history_lstm2_monthly.hdf5')
plot_model_architecture(model, output_model_path + 'model_lstm2_monthly.png')
plot_loss_curve(output_model_path, 'loss_plot_lstm2_monthly', train, valid)
plot_roc(output_model_path, 'roc_lstm2_monthly', fpr, tpr)
plot_pr(output_model_path, 'pr_lstm2_monthly', rec, pre)
###############################################





