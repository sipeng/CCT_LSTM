#*****************************
# This script implements the
# baseline model of neural
# network (mlp) using keras, 
# which is equivalent to
# slicing the observations
# into 12 pieces and conduct
# prediction at the end of
# each month
#
# Si Peng
# sipeng@adobe.com
# Aug. 01 2017
#*****************************


##### import modules ###########################
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

from utils import *
###############################################


##### load data ###############################
DATA_PATH = '/largedatadrive/dsnp/sipeng/data/sample/oneyear/'
#DATA_PATH = '/Users/sipeng/Downloads/output_si/' # for load developing
train_path = DATA_PATH + 'train.hdf5'
valid_path = DATA_PATH + 'valid.hdf5'
test_path  = DATA_PATH + 'test.hdf5'

print('############### Processing Data ####################')

x_train, y_train, sw_train = load_lr_baseline(train_path, 50)
x_valid, y_valid, sw_valid = load_lr_baseline(valid_path, 50)
x_test, y_test, sw_test    = load_lr_baseline(test_path, 50)

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

model.add(TimeDistributed(Dense(40, kernel_initializer='glorot_normal')))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(Dropout(0.5))

model.add(TimeDistributed(Dense(20, kernel_initializer='glorot_normal')))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(Dropout(0.5))

model.add(TimeDistributed(Dense(10, kernel_initializer='glorot_normal')))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(Dropout(0.5))

model.add(TimeDistributed(Dense(1, activation='sigmoid', kernel_initializer='glorot_normal')))
###############################################


##### model compiling #########################
# define some paths
cwd = os.getcwd()
cache_path = cwd+'/cache/'
output_model_path = cwd+'/model_output/'

# compile the model
#adam = Adam(lr=0.001, decay=1e-2)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'], sample_weight_mode = "temporal")

# create some callbacks
checkpointer = ModelCheckpoint(filepath=output_model_path+'bestmodel_mlp_monthly_baseline.hdf5', verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
###############################################


##### model fit ###############################
print(model.summary())

print('############### Fitting Model ####################')

History = model.fit(x_train, y_train, \
                    batch_size=50, epochs=200, shuffle=True, sample_weight=sw_train, \
                    validation_data = (x_valid, y_valid, sw_valid), \
                    initial_epoch = 0, verbose=2, callbacks=[checkpointer, earlystopper])

os.remove(cache_path + 'history_mlp_monthly_baseline.hdf5')
save_history(cache_path + 'history_mlp_monthly_baseline', History)
###############################################



##### model evaluation ########################

print('############### Evaluating Prediction ####################')

model  = load_model(output_model_path + 'bestmodel_mlp_monthly_baseline.hdf5')
y_pred = model.predict(x_test, verbose=1, batch_size = 50)

# extract the samples with weight = 1
y_pred1 = y_pred[y_test != -1]
y_test1 = y_test[y_test != -1]

# compute fpr and tpr
fpr, tpr, _ = roc_curve(y_test1, y_pred1)
auc_ROC     = auc(fpr, tpr)
pre, rec, _ = precision_recall_curve(y_test1, y_pred1)
auc_PR      = auc(rec, pre)
print(auc_ROC, auc_PR)

# plot the training and validating losses, and ROC curve
train, valid = load_history(cache_path + 'history_mlp_monthly_baseline.hdf5')
plot_model_architecture(model, output_model_path + 'model_mlp_monthly_baseline.png')
plot_loss_curve(output_model_path, 'loss_plot_mlp_monthly_baseline', train, valid)
plot_roc(output_model_path, 'roc_mlp_monthly_baseline', fpr, tpr)
plot_pr(output_model_path, 'pr_mlp_monthly_baseline', rec, pre)
###############################################

