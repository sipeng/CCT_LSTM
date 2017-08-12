#*****************************
# This script contains examples
# of training LSTM with output
# being a sequence, and studies
# the usage of masking
#
# Si Peng
# sipeng@adobe.com
# Jul. 14 2017
#*****************************

import pandas as pd
import numpy as np

#from tabulate import tabulate
from sklearn.metrics import roc_auc_score, matthews_corrcoef, \
precision_recall_fscore_support, average_precision_score, accuracy_score

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

#*****************************

## Example 1: many-to-many LSTM model

data_dim = 16
timesteps = 8
nb_classes = 10

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
# returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(TimeDistributed(Dense(nb_classes, activation='softmax')))  

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

# generate dummy training data
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, timesteps, nb_classes))

# generate dummy validation data
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, timesteps, nb_classes))

model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_val, y_val))

# model visualization
plot_model(model, to_file='model.png')


#*****************************

## Example 2: many-to-many LSTM model with a masking layer to deal with
## different input time lengths, note that the input are already padded

data_dim = 4
timesteps = 3
nb_classes = 2

model = Sequential()
model.add(Masking(mask_value=0., input_shape=(timesteps, data_dim))) 
model.add(LSTM(5, return_sequences=True))
model.add(TimeDistributed(Dense(nb_classes, activation='sigmoid'))) 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# generate dummy training data
x_train = np.random.random((6, timesteps, data_dim))
y_train = np.random.random((6, timesteps, nb_classes))

x_train[1, 1, :] = (0, )
y_train[1, 1, :] = (0, )

# generate dummy validation data
x_val = np.random.random((2, timesteps, data_dim))
y_val = np.random.random((2, timesteps, nb_classes))

model.fit(x_train, y_train, batch_size=2, epochs=5, validation_data=(x_val, y_val))

model.predict(x_train)
## note that the predictions at the masked timesteps are just a copy of the 
## predictions of the previous timestep, which is exactly what we expected

# model visualization
plot_model(model, to_file='model.png')


#*****************************

## Example 3. the same data input as Example 2, but use sample_weight
## to mask the output only, instead of using a masking layer

data_dim = 4
timesteps = 3
nb_classes = 2

model = Sequential()
#model.add(Masking(mask_value=0., input_shape=(timesteps, data_dim))) 
model.add(LSTM(5, input_shape=(timesteps, data_dim), return_sequences=True))
model.add(LSTM(5, return_sequences=True))
model.add(TimeDistributed(Dense(nb_classes, activation='sigmoid')))

model.compile(loss='binary_crossentropy', optimizer='adam', \
	sample_weight_mode = "temporal", metrics=['accuracy'])

print(model.summary())

# generate dummy training data
x_train = np.random.random((6, timesteps, data_dim))
y_train = np.random.random((6, timesteps, nb_classes))

x_train[1, 1, :] = (0, )
y_train[1, 1, :] = (0, )

# sample weight shoule be 2D in time series
s_weight = np.ones((6, timesteps), dtype=np.int)
s_weight[1, 1] = 0

# generate dummy validation data
x_val = np.random.random((2, timesteps, data_dim))
y_val = np.random.random((2, timesteps, nb_classes))

model.fit(x_train, y_train, batch_size=2, epochs=5, sample_weight=s_weight)

model.predict(x_train)

## as a comparison, the predictions at the 0-weight timestep are not the copy of the
## predictions at the previous timestep, meaning that the input features at the
## 0-weight timestep are not skipped, in fact they contribute to the predictions
## at future timesteps through the state of LSTM, and thus contribute to the loss

# model visualization
plot_model(model, to_file='model.png')
