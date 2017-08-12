#*****************************
# Process the E5 table, join
# with E32 and filter
# 
# Note: this script needs to
# be run under the current
# directory, instead of root
#
# Si Peng
# sipeng@adobe.com
# Jun. 26 2017
#*****************************

#***********************************
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark import HiveContext
from pyspark import SparkConf
import pyspark.sql.functions as F
from pyspark.sql.types import DateType, StringType
import glob, os, subprocess
#***********************************

#***********************************
sc = SparkContext()
sc.setLogLevel("WARN")
hvContext = HiveContext(sc)
sqlContext = SQLContext(sc)
#***********************************




#*********************************
# some utility functions
#********************************
def gene_product_dict(product_list):
	dict = {}
	for product in product_list:
		total = product.split(':')
		key = total[0].strip()
		values = eval(total[1])
		dict[key] = values
	return dict


# modifies the product names as their category name
def modify_product_name(product_dict, product_col):
	for key in product_dict.keys():
		if product_col in product_dict[key]:
			return key
	return "NA"

def udf_modify_product_name(product_dict):
	return F.udf(lambda c: modify_product_name(product_dict, c), StringType())



def process_X(df_X, df_Y, product_dict, product_interest, date_from = '2016-05-01', date_to = '2017-04-30'):
	# choose specific columns
	df_X = df_X['MEMBER_GUID', 'EVENT_TIME', 'PRODUCT_NAME', 'TOTAL_SESSION_TIME']
	# filter records between two dates
	dates = [date_from, date_to]
	date_from, date_to = [F.to_date(F.lit(s)).cast(DateType()) for s in dates]
	df_X = df_X.withColumn('EVENT_DATE', df_X['EVENT_TIME'].cast(DateType()))
	df_X = df_X.filter((F.col('EVENT_DATE') >= date_from) & (F.col('EVENT_DATE') <= date_to))
	# Change PRODUCT_NAME as the fisrt category and chose products of interest
	df_X = df_X.withColumn('PRODUCT_NAME', udf_modify_product_name(product_dict)(F.col("PRODUCT_NAME")))
	df_X = df_X[F.col('PRODUCT_NAME').isin(product_interest)]
	# remove duplicated rows
	df_X = df_X.dropDuplicates()
	## join with E32, filter the events
	df_XY = df_X.join(df_Y, (df_X.MEMBER_GUID == df_Y.MEMBER_GUID) & (df_X.EVENT_DATE >= df_Y.CREATE_DATE) \
		& (df_X.EVENT_TIME >= df_Y.START_TIME) & (df_X.EVENT_TIME <= df_Y.END_TIME), 'inner')\
	.drop(df_Y.MEMBER_GUID).drop(df_X.EVENT_TIME).drop(df_Y.START_TIME).drop(df_Y.END_TIME)
	return df_XY




#**************************
# process E5 data
#**************************
total_path = 'wasb://cccm@adobedatascience.blob.core.windows.net/full_data_parquet_incremental/5/'
months = ['2016-10', '2016-11', '2016-12', '2017-01', '2017-02', '2017-03', '2017-04']
event5_path = [total_path + m for m in months]

y_path = 'wasb://cccm@adobedatascience.blob.core.windows.net/cct_lstm/32/'
output_path = 'wasb://cccm@adobedatascience.blob.core.windows.net/cct_lstm/5/'

product_name_path = './product_name.txt'
product_interest = ['Photoshop', 'Illustrator', 'InDesign', 'PremierePro', 'Lightroom', 'AfterEffects', \
'MediaEncoder', 'DreamWeaver']


# clear the output_path
cmd = ['hdfs', 'dfs', '-rm', '-r', output_path]
subprocess.call(cmd)


# read product list
with open(product_name_path) as f:
	product_list = f.readlines()

product_dict = gene_product_dict(product_list)



# read and process E5 data
df_X = hvContext.read.parquet(event5_path[0])
df_Y = hvContext.read.parquet(y_path)
df_XY = process_X(df_X, df_Y, product_dict, product_interest)

# combine all the processed data in E5
for path in event5_path[1:]:
	df_path = hvContext.read.parquet(path)
	df_path = process_X(df_path, df_Y, product_dict, product_interest)
	df_XY = df_XY.unionAll(df_path)

# write the data
df_XY.write.parquet(output_path)
sc.stop()






"""
## play with the data
test_file_path = total_path + '2016-11/part-r-00000-1e45d2ce-97a2-46d6-aa86-c60c4376c137.gz.parquet'
test_file_y_path = y_path + 'part-r-00000-e909a36a-a626-4008-b2f2-7ee24a5f3472.gz.parquet'
df_X = hvContext.read.parquet(test_file_path)
df_Y = hvContext.read.parquet(y_path)

## check output dimension
path = 'wasb://cccm@adobedatascience.blob.core.windows.net/cct_lstm/5/'
df = hvContext.read.parquet(path)
df.show()
df.count() # 98933810
df.select('MEMBER_GUID').distinct().count() # 683,924
df.select('SEAT_ID').distinct().count() # 732,819
df.select('CONTRACT_ID').distinct().count() # 267,998
"""
