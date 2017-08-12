#*****************************
# This file implements the
# LSTM2 of the hierarchical
# LSTM model, the training
# and prediction are at 
# contract level
#
# Si Peng
# sipeng@adobe.com
# Jul. 22 2017
#*****************************


##### import modules ###########################
import pandas as pd
import numpy as np
import h5py

from tabulate import tabulate
from sklearn.metrics import roc_auc_score, matthews_corrcoef, \
precision_recall_fscore_support, average_precision_score, \
roc_curve, auc, accuracy_score

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

from utils import *
###############################################


##### load data ###############################
cwd = os.getcwd()
input_path = cwd + '/data_input/'

x_train  = np.load(input_path + 'train_lstm2.npy')
x_valid  = np.load(input_path + 'valid_lstm2.npy')
x_test   = np.load(input_path + 'test_lstm2.npy')

y_train  = np.load(input_path + 'y_train_lstm2.npy')
y_valid  = np.load(input_path + 'y_valid_lstm2.npy')
y_test   = np.load(input_path + 'y_test_lstm2.npy')

sw_train = np.load(input_path + 'sw_train_lstm2.npy')
sw_valid = np.load(input_path + 'sw_valid_lstm2.npy')
sw_test  = np.load(input_path + 'sw_test_lstm2.npy')
###############################################



##### model building **************************
model = Sequential()
model.add(Masking(mask_value=-1.0, input_shape=(x_train.shape[1], x_train.shape[2]))) 

model.add(LSTM(20, return_sequences=True))
model.add(Dropout(0.3))

model.add(TimeDistributed(Dense(1, activation='sigmoid')))
###############################################


##### model compiling #########################
# define some paths
cwd = os.getcwd()
cache_path = cwd+'/cache/'
output_model_path = cwd+'/model_output/'

# compile the model
adam = Adam(lr=0.01, decay=1e-02)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'], sample_weight_mode = "temporal")

# create some callbacks
checkpointer = ModelCheckpoint(filepath=output_model_path+'bestmodel_lstm2.hdf5', verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
###############################################


##### model fit ###############################
print(model.summary())

History = model.fit(x_train, y_train, sample_weight = sw_train, \
                    batch_size=200, epochs=10, shuffle=True, \
                    validation_data = (x_valid, y_valid, sw_valid), \
                    initial_epoch = 0, verbose=1, callbacks=[checkpointer, earlystopper])

save_history(cache_path + 'history_lstm2', History)
###############################################



##### model evaluation ########################
model  = load_model(output_model_path + 'bestmodel_lstm2.hdf5')
y_pred = model.predict(x_test, verbose=1, batch_size = 200)

# extract the samples with weight = 1
y_pred1 = y_pred[sw_test != 0.]
y_test1 = y_test[sw_test != 0.]

# compute fpr and tpr
fpr, tpr, _ = roc_curve(y_test1, y_pred1)
auc_ROC     = auc(fpr, tpr)

# plot the training and validating losses, and ROC curve
train, valid = load_history(cache_path + 'history_lstm2.hdf5')
plot_loss_curve(output_model_path + 'loss_plot_lstm2.png', train, valid)
plot_roc(output_model_path + 'roc_lstm2.png', fpr, tpr)
###############################################





