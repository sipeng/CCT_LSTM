#*****************************
# Pad the nulls in the samples
# using a specified value,
# then save the samples in
# hdf5 compressed file format
#
# Si Peng
# sipeng@adobe.com
# Jul. 12 2017
#*****************************

#***********************************
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark import HiveContext
from pyspark import SparkConf
import pyspark.sql.functions as F
from pyspark.sql.types import DateType, ArrayType, IntegerType, LongType
from datetime import datetime, date
import numpy as np
import pandas as pd
import glob, os, subprocess
import h5py
#***********************************

#***********************************
sc = SparkContext()
sc.setLogLevel("ERROR")
hvContext = HiveContext(sc)
sqlContext = SQLContext(sc)
#***********************************



#*************define some functions**********************
def get_file_names(path_dir):
	cmd = ['hdfs', 'dfs', '-find', path_dir, '-name', '*.parquet']
	files = subprocess.check_output(cmd).strip().split('\n')
  	return files


def parquet_to_npArray(df_input):
	df_np = df_input.toPandas().as_matrix()
	# extract seat id and contract id
	np_sid = np.array(df_np[:, -10]).astype(str)
	np_cid = np.array(df_np[:, -9]).astype(str)
	# extract labels
	np_seat_daily       = np.array(df_np[:, -8].tolist())
	np_seat_monthly     = np.array(df_np[:, -7].tolist())
	np_contract_daily   = np.array(df_np[:, -4].tolist())
	np_contract_monthly = np.array(df_np[:, -3].tolist())
	# extract create and cancel days
	np_seat_create     = np.array(df_np[:, -6]).astype(int)
	np_seat_cancel     = np.array(df_np[:, -5]).astype(int)
	np_contract_create = np.array(df_np[:, -2]).astype(int)
	np_contract_cancel = np.array(df_np[:, -1]).astype(int)
	# extract features, replace None with -1
	np_feature = np.array(df_np[:, :-10].tolist())
	np_feature = np.where(np_feature == np.array(None), -1, np_feature)
	np_feature = np_feature.astype(int)
	return np_sid, np_cid, np_feature, np_seat_daily, np_seat_monthly, np_seat_create, np_seat_cancel, \
	np_contract_daily, np_contract_monthly, np_contract_create, np_contract_cancel


def npArray_to_hdf5(np_sid, np_cid, np_feature, np_seat_daily, np_seat_monthly, np_seat_create, np_seat_cancel, \
	np_contract_daily, np_contract_monthly, np_contract_create, np_contract_cancelc, file_name, output_path):
	file_name = file_name.split('/')[-1] + '.hdf5'
	file_name = output_path + file_name
	h5f = h5py.File(file_name, 'w')
	h5f.create_dataset('seat_id', data = np_sid, compression="gzip", compression_opts=9)
	h5f.create_dataset('contract_id', data = np_cid, compression="gzip", compression_opts=9)
	h5f.create_dataset('X', data = np_feature, compression="gzip", compression_opts=9)
	h5f.create_dataset('label_seat_daily', data = np_seat_daily, compression="gzip", compression_opts=9)
	h5f.create_dataset('label_seat_monthly', data = np_seat_monthly, compression="gzip", compression_opts=9)
	h5f.create_dataset('seat_create', data = np_seat_create, compression="gzip", compression_opts=9)
	h5f.create_dataset('seat_cancel', data = np_seat_cancel, compression="gzip", compression_opts=9)
	h5f.create_dataset('label_contract_daily', data = np_contract_daily, compression="gzip", compression_opts=9)
	h5f.create_dataset('label_contract_monthly', data = np_contract_monthly, compression="gzip", compression_opts=9)
	h5f.create_dataset('contract_create', data = np_contract_create, compression="gzip", compression_opts=9)
	h5f.create_dataset('contract_cancel', data = np_contract_cancel, compression="gzip", compression_opts=9)
	h5f.close()


#**************************
# process data
#**************************

input_path = 'wasb://cccm@adobedatascience.blob.core.windows.net/cct_lstm/sample_1/'
output_path = '/mnt/cccm/output_si/'

# clear the output_path
cmd = ['rm', '-f', output_path+'*']
subprocess.call(cmd)


file_names = get_file_names(input_path)

for file_name in file_names:
	print file_name.split('/')[-1]
	df_parquet = hvContext.read.parquet(file_name)
	np_sid, np_cid, np_feature, np_seat_daily, np_seat_monthly, np_seat_create, np_seat_cancel, \
	np_contract_daily, np_contract_monthly, np_contract_create, np_contract_cancel = parquet_to_npArray(df_parquet)

	npArray_to_hdf5(np_sid, np_cid, np_feature, np_seat_daily, np_seat_monthly, np_seat_create, np_seat_cancel, \
		np_contract_daily, np_contract_monthly, np_contract_create, np_contract_cancel, file_name, output_path)
	print np_sid.shape


sc.stop()




"""
#**************************
## dev, use a small sample
#**************************
input_path = 'wasb://cccm@adobedatascience.blob.core.windows.net/cct_lstm/sample_1/'
output_path = '/mnt/cccm/output_si/'
test_path = input_path + 'part-r-00000*'
df1 = hvContext.read.parquet(test_path)

np_sid, np_cid, np_feature, np_seat_daily, np_seat_monthly, np_seat_create, np_seat_cancel, \
np_contract_daily, np_contract_monthly, np_contract_create, np_contract_cancel = parquet_to_npArray(df_parquet)

npArray_to_hdf5(np_sid, np_cid, np_feature, np_seat_daily, np_seat_monthly, np_seat_create, np_seat_cancel, \
	np_contract_daily, np_contract_monthly, np_contract_create, np_contract_cancel, file_name, output_path)

df_np = df1.toPandas().as_matrix()
# extract seat id
np_sid = np.array(df_np[:, -6])
np_sid = np_sid.astype(str)
# extract contract id
np_cid = np.array(df_np[:, -5])
np_cid = np_cid.astype(str)
# extract labels
np_seat_daily = df_np[:, -4]
np_seat_daily = np.array(np_seat_daily.tolist())
np_seat_monthly = df_np[:, -3]
np_seat_monthly = np.array(np_seat_monthly.tolist())
np_contract_daily = df_np[:, -2]
np_contract_daily = np.array(np_contract_daily.tolist())
np_contract_monthly = df_np[:, -1]
np_contract_monthly = np.array(np_contract_monthly.tolist())

# extract features, replace None with -1
np_feature = df_np[:, :-6]
np_feature = np.array(np_feature.tolist())
np_feature = np.where(np_feature == np.array(None), -1, np_feature)
np_feature = np_feature.astype(int)

# write to hdf5 files
file_names = get_file_names(input_path)
npArray_to_hdf5(np_sid, np_cid, np_feature, np_seat_daily, np_seat_monthly, np_contract_daily, np_contract_monthly, file_names[0], output_path)

file_name = file_names[0]
file_name = file_name.split('/')[-1] + '.hdf5'
file_name = output_path + file_name
h5f = h5py.File(file_name, 'w')
h5f.create_dataset('seat_id', data = np_sid, compression="gzip", compression_opts=9)
h5f.create_dataset('contract_id', data = np_cid, compression="gzip", compression_opts=9)
h5f.create_dataset('X', data = np_feature, compression="gzip", compression_opts=9)
h5f.create_dataset('label_seat_daily', data = np_seat_daily, compression="gzip", compression_opts=9)
h5f.create_dataset('label_seat_monthly', data = np_seat_monthly, compression="gzip", compression_opts=9)
h5f.create_dataset('label_contract_daily', data = np_contract_daily, compression="gzip", compression_opts=9)
h5f.create_dataset('label_contract_monthly', data = np_contract_monthly, compression="gzip", compression_opts=9)
h5f.close()

# check if the files are written correctly
data_path = output + ''
data = h5py.File(output_path)
data_np = np.array(data)
"""
