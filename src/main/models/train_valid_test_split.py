#*****************************
# Split train/valid/test set 
# from the original data
#
# Si Peng
# sipeng@adobe.com
# Jul. 25 2017
#*****************************

##### import modules ##########################
import pandas as pd
import numpy as np
import h5py
import timeit
import os
from utils import *
###############################################



##### read in data ############################
#DATA_PATH = '/largedatadrive/dsnp/sipeng/data/full/oneyear/'
DATA_PATH = '/Users/sipeng/Downloads/output_si/' # for load developing

# get a list of all file names
files = os.listdir(DATA_PATH)
files = [i for i in files if i.endswith('.hdf5')]

# read all the c_ids
file = files[0]
data_block  = h5py.File(DATA_PATH + file)
contract_id = np.array(data_block['contract_id'])
data_block.close()

for file in files[1:]:
	data_block = h5py.File(DATA_PATH + file)
	contract_id = np.concatenate((contract_id, np.array(data_block['contract_id'])))
	data_block.close()

uniq_cid = np.unique(contract_id)

# sample 10,000 for train, 3,000 each for valid/test
index_list = np.random.choice(len(uniq_cid), 16000, replace=False)
train_cid  = uniq_cid[index_list[0:10000]]
valid_cid  = uniq_cid[index_list[10000:13000]]
test_cid   = uniq_cid[index_list[13000:]]
###############################################




##### split the sets and save the data ########
seat_id, contract_id, X, y1_daily, y1_monthly, seat_create, seat_cancel, \
y2_daily, y2_monthly, contract_create, contract_cancel = split_data(DATA_PATH, files, train_cid)

npArray_to_hdf5(seat_id, contract_id, X, y1_daily, y1_monthly, seat_create, seat_cancel, \
	y2_daily, y2_monthly, contract_create, contract_cancel, 'train', DATA_PATH)

seat_id, contract_id, X, y1_daily, y1_monthly, seat_create, seat_cancel, \
y2_daily, y2_monthly, contract_create, contract_cancel = split_data(DATA_PATH, files, valid_cid)

npArray_to_hdf5(seat_id, contract_id, X, y1_daily, y1_monthly, seat_create, seat_cancel, \
	y2_daily, y2_monthly, contract_create, contract_cancel, 'valid', DATA_PATH)

seat_id, contract_id, X, y1_daily, y1_monthly, seat_create, seat_cancel, \
y2_daily, y2_monthly, contract_create, contract_cancel = split_data(DATA_PATH, files, test_cid)

npArray_to_hdf5(seat_id, contract_id, X, y1_daily, y1_monthly, seat_create, seat_cancel, \
	y2_daily, y2_monthly, contract_create, contract_cancel, 'test', DATA_PATH)
###############################################

