import tensorflow as tf
import os
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql.types import *
from pyspark.sql import DataFrame
from pyspark.sql.functions import *


# Use this function to normalize, crop and resize images.
def tf_norm_crop_resize_image(image, resize_dim = (224, 224)):
	"""Normalizes image to [0.,1.], crops to dims (64, 64, 3)
	and resizes to `resize_dim`, returning an image tensor."""
	image = tf.cast(image, tf.float32)/255.
	image = tf.image.resize_with_crop_or_pad(image, 112, 112)
	return image


def one_hot_encode(label):
	return tf.one_hot(label, depth=1000).numpy().tolist()


def apply_modifications():
	os.chdir(data_dir)
	schema = StructType([StructField('label', IntegerType(), True),
                     StructField('file_name', StringType(), True),
                     StructField('image', BinaryType(), True)])
	for f in os.listdir():
		df = sqlContext.read.schema(schema).format("tfrecords").option("recordType", "SequenceExample").load(os.path.join(data_dir, f))
		df = df.select("image", "label")
		df = df.withColumn("label", udf_one_hot("label")).toPandas()
		df['image'].apply(lambda image: tf_norm_crop_resize_image(image))
		df = spark.createDataFrame(df)
		df.write.schema(schema).format("tfrecords").option("recordType", "SequenceExample").save(os.path.join(data_dir_spark, f))


if __name__=='__main__':
	working_dir = os.getcwd()
	data_dir = 'data/imagenet/imagenet2012/5.1.0/'
	data_dir_spark = 'data/imagenet/imagenet2012/5.1.0_spark/'
	conf = SparkConf().setMaster('local[*]').setAppName('Data_wrangling')
	sc = SparkContext(conf = conf)
	sqlContext = SQLContext(sc)
	udf_one_hot = udf_pandas(one_hot_encode, ArrayType(FloatType()))
	udf_function = udf_pandas(tf_norm_crop_resize_image, ArrayType(FloatType()))
	apply_modifications()
