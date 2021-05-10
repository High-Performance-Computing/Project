import tensorflow as tf
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql.types import *
import os 
from functools import reduce  # For Python 3.x
from pyspark.sql import DataFrame
import time
import tensorflow as tf
from pyspark.sql.functions import *


working_dir = os.getcwd()
data_dir = 'data/imagenet/imagenet2012/5.1.0/'
conf = SparkConf().setMaster('local[*]').setAppName('Data_wrangling')
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

os.chdir(data_dir)
schema = StructType([StructField('label', IntegerType(), True),
                     StructField('file_name', StringType(), True),
                     StructField('image', BinaryType(), True)])
for f in os.listdir()[:1]:
	df = sqlContext.read.schema(schema).format("tfrecords").option("recordType", "SequenceExample").load(os.path.join(data_dir, f))

print('Congratulations, your environment is set up')
