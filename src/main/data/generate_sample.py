#*****************************
# Generate samples using
# joined and filtered table
#
# Si Peng
# sipeng@adobe.com
# Jun. 29 2017
#*****************************

#***********************************
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark import HiveContext
from pyspark import SparkConf
import pyspark.sql.functions as F
from pyspark.sql.types import DateType, StringType, ArrayType, IntegerType
from datetime import datetime, date
import glob, os, subprocess
#***********************************

#***********************************
sc = SparkContext()
sc.setLogLevel("ERROR")
hvContext = HiveContext(sc)
sqlContext = SQLContext(sc)
#***********************************


#***********************************
# some functions
def gene_exprs(product_feature, event_feature, product_list):
	expr_total_time = [F.sum(F.when(F.col(product_feature).isin(p), F.col(event_feature)).otherwise(F.lit(0))) for p in product_list]
	expr_count = [F.sum(F.when(F.col(product_feature).isin(p), F.lit(1)).otherwise(F.lit(0))) for p in product_list]
	exprs = expr_total_time + expr_count
	return exprs

## udf 1, create daily churn labels
def gene_churn_label_daily(create_day, churn_day, output_dim):
	multi_label = [-1]*output_dim
	if churn_day < output_dim:
		multi_label[create_day:churn_day] = [0]*(churn_day-create_day)
		multi_label[churn_day] = 1
	else:
		multi_label[create_day:] = [0]*(output_dim-create_day)
	return multi_label

def udf_gene_churn_label_daily(output_dim):
	return F.udf(lambda c1, c2: gene_churn_label_daily(c1, c2, output_dim), ArrayType(IntegerType()))



## udf2, create montly labels
def gene_churn_label_montly(create_day, churn_day, output_dim):
	multi_label = [-1]*output_dim
	n_create = int(create_day/30)
	n_churn = int(churn_day/30)
	if n_churn < output_dim:
		multi_label[n_create:n_churn] = [0]*(n_churn-n_create)
		multi_label[n_churn] = 1
	else:
		multi_label[n_create:] = [0]*(output_dim-n_create)
	return multi_label

def udf_gene_churn_label_montly(output_dim):
	return F.udf(lambda c1, c2: gene_churn_label_montly(c1, c2, output_dim), ArrayType(IntegerType()))



## create the feature tensor, as well as the churn labels at seat level and contract level
def generate_sample(df, date_from = "2016-05-01", date_to = "2017-04-30", date_to_cancel = '2017-05-31'):
	# generte time steps and date_to_cancel
	date_from = datetime.strptime(date_from, '%Y-%m-%d').date()
	date_to = datetime.strptime(date_to, '%Y-%m-%d').date()
	max_days = (date_to - date_from).days
	time_steps = range(0, max_days+1)
	date_to_cancel = datetime.strptime(date_to_cancel, '%Y-%m-%d').date()
	# create a constant column of day0
	day0 = df.select(F.min(df.CREATE_DATE)).collect()[0][0] # day0
	df = df.withColumn('DAY0', F.lit(day0))
	## step 1. generate feature
	df1 = df.withColumn('AGE', F.datediff('EVENT_DATE', 'DAY0'))
	# pivot all samples and return different time steps
	all_products = [str(p.PRODUCT_NAME) for p in df1.select('PRODUCT_NAME').distinct().collect()]
	exprs = gene_exprs(product_feature='PRODUCT_NAME', event_feature='TOTAL_SESSION_TIME', product_list=all_products)
	df1 = df1.groupby('SEAT_ID').pivot('AGE', time_steps).agg(F.array(*exprs))
	## step 2. generate seat level churn labels: montly and daily labels are both generated
	# group by seat_id
	df2 = df.groupby('SEAT_ID').agg(F.min('CREATE_DATE').alias('SEAT_CREATE_DATE'), \
		F.min('SEAT_CANCELLED_DATE').alias('SEAT_CANCELLED_DATE'), F.min('CONTRACT_ID').alias('CONTRACT_ID'), \
		F.min('DAY0').alias('DAY0'))
	# create days columns
	df2 = df2.withColumn('SEAT_CREATE_DAYS', F.datediff('SEAT_CREATE_DATE', 'DAY0')).drop('SEAT_CREATE_DATE')
	df2 = df2.withColumn('SEAT_CANCELLED_DAYS', F.datediff('SEAT_CANCELLED_DATE', 'DAY0')).drop('SEAT_CANCELLED_DATE')
	# generate labels
	df2 = df2.withColumn('SEAT_LABEL_DAILY', udf_gene_churn_label_daily(max_days+1)(F.col('SEAT_CREATE_DAYS'), F.col('SEAT_CANCELLED_DAYS')))
	df2 = df2.withColumn('SEAT_LABEL_MONTHLY', udf_gene_churn_label_montly(13)(F.col('SEAT_CREATE_DAYS'), F.col('SEAT_CANCELLED_DAYS')))
	df2 = df2.select('SEAT_ID', 'CONTRACT_ID', 'SEAT_LABEL_DAILY', 'SEAT_LABEL_MONTHLY', 'SEAT_CREATE_DAYS', 'SEAT_CANCELLED_DAYS')
	## step 3. generate contract level churn labels: montly and daily labels are both generated
	# group by contract_id
	df3 = df.groupby('CONTRACT_ID').agg(F.min('CREATE_DATE').alias('CONTRACT_CREATE_DATE'), \
		F.min('CONTRACT_CANCELLED_DATE').alias('CONTRACT_CANCELLED_DATE'), F.min('DAY0').alias('DAY0'))
	# create days columns
	df3 = df3.withColumn('CONTRACT_CREATE_DAYS', F.datediff('CONTRACT_CREATE_DATE', 'DAY0')).drop('CONTRACT_CREATE_DATE')
	df3 = df3.withColumn('CONTRACT_CANCELLED_DAYS', F.datediff('CONTRACT_CANCELLED_DATE', 'DAY0')).drop('CONTRACT_CANCELLED_DATE')
	# generate labels
	df3 = df3.withColumn('CONTRACT_LABEL_DAILY', udf_gene_churn_label_daily(max_days+1)(F.col('CONTRACT_CREATE_DAYS'), F.col('CONTRACT_CANCELLED_DAYS')))
	df3 = df3.withColumn('CONTRACT_LABEL_MONTHLY', udf_gene_churn_label_montly(13)(F.col('CONTRACT_CREATE_DAYS'), F.col('CONTRACT_CANCELLED_DAYS')))
	df3 = df3.select('CONTRACT_ID', 'CONTRACT_LABEL_DAILY', 'CONTRACT_LABEL_MONTHLY', 'CONTRACT_CREATE_DAYS', 'CONTRACT_CANCELLED_DAYS')
	## join the tables
	df2 = df2.join(df3, df2.CONTRACT_ID==df3.CONTRACT_ID).drop(df3.CONTRACT_ID)
	df1 = df1.join(df2, df1.SEAT_ID==df2.SEAT_ID).drop(df2.SEAT_ID)
	return df1







#**************************
# process data
#**************************
input_path = 'wasb://cccm@adobedatascience.blob.core.windows.net/cct_lstm/5/'
output_path = 'wasb://cccm@adobedatascience.blob.core.windows.net/cct_lstm/sample_1/'

# clear the output_path
cmd = ['hdfs', 'dfs', '-rm', '-r', output_path]
subprocess.call(cmd)

# read and process all the data
df = hvContext.read.parquet(input_path)
df = generate_sample(df)

# write the data
df.write.parquet(output_path)
sc.stop()








"""
def get_file_names(path_dir):
	cmd = ['hdfs', 'dfs', '-find', path_dir, '-name', '*.parquet']
	files = subprocess.check_output(cmd).strip().split('\n')
  	return files


file_names = get_file_names(input_path)

df = hvContext.read.parquet(file_names[0])

df0 = df.withColumn('CHECK', F.when(F.col('EVENT_DATE')>=F.col('CREATE_DATE'), 0).otherwise(1))
df0.agg(F.sum('CHECK')).show()

## play with the data
df_contract = df.groupby('SEAT_ID').agg(F.min('CONTRACT_ID').alias('MIN'), F.max('CONTRACT_ID').alias('MAX'))
df_contract = df_contract.withColumn('CHECK', F.when(F.col('MIN')==F.col('MAX'), 0).otherwise(1))
df_contract.agg(F.sum('CHECK')).show()  # 0, meaning no re-used SEAT_ID across CONTRACTs


df1 = df.groupby('SEAT_ID').agg(F.min('CREATE_DATE').alias('MIN'), F.max('CREATE_DATE').alias('MAX'))
df1 = df1.withColumn('CHECK', F.when(F.col('MIN')==F.col('MAX'), 0).otherwise(1))
df1.agg(F.sum('CHECK')).show()  # 0, meaning each seat only has one created date

df2 = df.groupby('SEAT_ID').agg(F.min('SEAT_CANCELLED_DATE').alias('MIN'), F.max('SEAT_CANCELLED_DATE').alias('MAX'))
df2 = df2.withColumn('CHECK', F.when(F.col('MIN')==F.col('MAX'), 0).otherwise(1))
df2.agg(F.sum('CHECK')).show()  # 0, meaning each seat only has one cancelled date

df3 = df.groupby('SEAT_ID').agg(F.min('MEMBER_GUID').alias('MIN'), F.max('MEMBER_GUID').alias('MAX'))
df3 = df3.withColumn('CHECK', F.when(F.col('MIN')==F.col('MAX'), 0).otherwise(1))
df3.agg(F.sum('CHECK')).show()  # not 0, meaning a seat can be assigned to multiple members during its lifetime


###########################################
## generate df1, df2 and df3
###########################################

# prepare the data, create day0 column
day0 = df.select(F.min(df.CREATE_DATE)).collect()[0][0] # day0
df = df.withColumn('DAY0', F.lit(day0))

###########################################
## generate df1
###########################################

df1 = df.withColumn('AGE', F.datediff('EVENT_DATE', 'DAY0'))

date_from = "2016-05-01"
date_to = "2017-04-30"
date_to_cancel = '2017-05-31'
date_from = datetime.strptime(date_from, '%Y-%m-%d').date()
date_to = datetime.strptime(date_to, '%Y-%m-%d').date()
max_days = (date_to - date_from).days
time_steps = range(0, max_days+1)
date_to_cancel = datetime.strptime(date_to_cancel, '%Y-%m-%d').date()

# pivot all samples and return different time steps
all_products = [str(p.PRODUCT_NAME) for p in df1.select('PRODUCT_NAME').distinct().collect()]
exprs = gene_exprs(product_feature='PRODUCT_NAME', event_feature='TOTAL_SESSION_TIME', product_list=all_products)
df1 = df1.groupby('SEAT_ID').pivot('AGE', time_steps).agg(F.array(*exprs))

###########################################
## generate df2
###########################################

# group by seat_id
df2 = df.groupby('SEAT_ID').agg(F.min('CREATE_DATE').alias('SEAT_CREATE_DATE'), \
	F.min('SEAT_CANCELLED_DATE').alias('SEAT_CANCELLED_DATE'), F.min('CONTRACT_ID').alias('CONTRACT_ID'), \
	F.min('DAY0').alias('DAY0'))
# create days columns
df2 = df2.withColumn('SEAT_CREATE_DAYS', F.datediff('SEAT_CREATE_DATE', 'DAY0')).drop('SEAT_CREATE_DATE')
df2 = df2.withColumn('SEAT_CANCELLED_DAYS', F.datediff('SEAT_CANCELLED_DATE', 'DAY0')).drop('SEAT_CANCELLED_DATE')
# generate labels
df2 = df2.withColumn('SEAT_LABEL_DAILY', udf_gene_churn_label_daily(max_days+1)(F.col('SEAT_CREATE_DAYS'), F.col('SEAT_CANCELLED_DAYS')))
df2 = df2.withColumn('SEAT_LABEL_MONTHLY', udf_gene_churn_label_montly(13)(F.col('SEAT_CREATE_DAYS'), F.col('SEAT_CANCELLED_DAYS')))
df2 = df2.select('SEAT_ID', 'CONTRACT_ID', 'SEAT_LABEL_DAILY', 'SEAT_LABEL_MONTHLY')

# check
df2.filter("SEAT_ID='0079571727DA1214D20A'").show()
df01 = df1.filter("SEAT_ID='0079571727DA1214D20A'")
df01.agg(F.max('AGE')).show()



# for debugging
df22 = df2.withColumn('SEAT_LABEL_MONTHLY', udf_gene_churn_label_montly(13)(F.col('SEAT_CREATE_DAYS'), F.col('SEAT_CANCELLED_DAYS')))
df22.orderBy('SEAT_CREATE_DAYS', 'SEAT_CANCELLED_DAYS').show()
df22.filter("SEAT_ID='C42BF2873D95D3114B9A'").collect()

###########################################
## generate df3
###########################################

# group by contract_id
df3 = df.groupby('CONTRACT_ID').agg(F.min('CREATE_DATE').alias('CONTRACT_CREATE_DATE'), \
	F.min('CONTRACT_CANCELLED_DATE').alias('CONTRACT_CANCELLED_DATE'), F.min('DAY0').alias('DAY0'))
# create days columns
df3 = df3.withColumn('CONTRACT_CREATE_DAYS', F.datediff('CONTRACT_CREATE_DATE', 'DAY0')).drop('CONTRACT_CREATE_DATE')
df3 = df3.withColumn('CONTRACT_CANCELLED_DAYS', F.datediff('CONTRACT_CANCELLED_DATE', 'DAY0')).drop('CONTRACT_CANCELLED_DATE')
# generate labels
df3 = df3.withColumn('CONTRACT_LABEL_DAILY', udf_gene_churn_label_daily(max_days+1)(F.col('CONTRACT_CREATE_DAYS'), F.col('CONTRACT_CANCELLED_DAYS')))
df3 = df3.withColumn('CONTRACT_LABEL_MONTHLY', udf_gene_churn_label_montly(13)(F.col('CONTRACT_CREATE_DAYS'), F.col('CONTRACT_CANCELLED_DAYS')))
df3 = df3.select('CONTRACT_ID', 'CONTRACT_LABEL_DAILY', 'CONTRACT_LABEL_MONTHLY')

df23 = df2.join(df3, df2.CONTRACT_ID==df3.CONTRACT_ID).drop(df3.CONTRACT_ID)

import org.apache.spark.util.SizeEstimator
print(SizeEstimator.estimate(df))
"""

