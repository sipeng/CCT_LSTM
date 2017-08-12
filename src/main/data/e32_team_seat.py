#*****************************
# Process the E32 table
#
# Si Peng
# sipeng@adobe.com
# Jun. 27 2017
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


#***********************************

def process_Y(df_Y, date_from = '2016-05-01', date_to = '2017-05-30'):
	# choose needed columns
	df_Y = df_Y['MEMBER_GUID', 'CONTRACT_ID', 'SEAT_ID', 'START_TIME', 'END_TIME', 'SEAT_CREATED_DATE', 'SEAT_CANCELLED_DATE', 'CONTRACT_CREATED_TIME', 'CONTRACT_CANCELLED_DATE']
	# filter out the unassigned sessions
	df_Y = df_Y.filter(df_Y.MEMBER_GUID != 'GUID_IS_NULL_FOR_THE_ROW')
	# filter out seat sessions beyond the cancellation date
	df_Y = df_Y.filter(df_Y.END_TIME.cast(DateType()) <= df_Y.SEAT_CANCELLED_DATE)
	# filter records between two dates
	dates = [date_from, date_to]
	date_from, date_to = [F.to_date(F.lit(s)).cast(DateType()) for s in dates]
	df_Y = df_Y.withColumn('CREATE_DATE', F.greatest('SEAT_CREATED_DATE', 'CONTRACT_CREATED_TIME').cast(DateType()))
	df_Y = df_Y.filter((F.col('CREATE_DATE') >= date_from) & (F.col('CREATE_DATE') <= date_to))
	# remove unneeded columns
	df_Y = df_Y.drop('SEAT_CREATED_DATE').drop('CONTRACT_CREATED_TIME')
	# cast into dates
	df_Y = df_Y.withColumn('SEAT_CANCELLED_DATE', df_Y['SEAT_CANCELLED_DATE'].cast(DateType()))
	df_Y = df_Y.withColumn('CONTRACT_CANCELLED_DATE', df_Y['CONTRACT_CANCELLED_DATE'].cast(DateType()))
	df_Y = df_Y.dropDuplicates()
	return df_Y



#***********************************


#**************************
# process E32 data
#**************************
total_path = 'wasb://cccm@adobedatascience.blob.core.windows.net/cct/2017-05-30/TEAM_SEAT_EVENTS_JOIN_2017-05-30/'
output_path = 'wasb://cccm@adobedatascience.blob.core.windows.net/cct_lstm/32/'

# clear the output_path
cmd = ['hdfs', 'dfs', '-rm', '-r', output_path]
subprocess.call(cmd)

# read and process all the data
df_Y = hvContext.read.parquet(total_path)
df_Y = process_Y(df_Y)

# write the data
df_Y.write.parquet(output_path)
sc.stop()








"""
total_path = 'wasb://cccm@adobedatascience.blob.core.windows.net/cct/2017-05-30/TEAM_SEAT_EVENTS_JOIN_2017-05-30/'
df_Y = hvContext.read.parquet(total_path)
df_Y = df_Y.filter("SEAT_ID='0079571727DA1214D20A'")
df_Y = df_Y['MEMBER_GUID', 'SEAT_ID', 'START_TIME', 'END_TIME', 'SEAT_CREATED_DATE', 'SEAT_CANCELLED_DATE']
df_Y.show()
df_Y.filter(df_Y.END_TIME <= df_Y.SEAT_CANCELLED_DATE).show()

## play with the data
test_file_path = total_path + 'part-r-00000-43e865b7-0ed0-4655-9079-3984210aa1a7.gz.parquet'

df_Y = hvContext.read.parquet(test_file_path)
df_Y = process_Y(df_Y)

test_file_path = output_path + 'part-r-00000-e909a36a-a626-4008-b2f2-7ee24a5f3472.gz.parquet'
df_Y = hvContext.read.parquet(test_file_path)

df_Y = hvContext.read.parquet(output_path)

df_Y.select('MEMBER_GUID').distinct().count() #1,148,713
df_Y.select('SEAT_ID').distinct().count() #1,182,176

"""
