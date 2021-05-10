import os
import tensorflow_datasets as tfds
import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np
from IPython.display import Image
from keras.optimizers import Adam
import tensorflow as tf
import time

working_dir = '/n/holyscratch01/Academic-cluster/Spring_2021/g_84102/SCRATCH/ImageNet'
AUTOTUNE = tf.data.AUTOTUNE



def load_data(batch_size, spark):
    if not spark:
    	[ds_train, ds_val], ds_info = tfds.load('imagenet2012', split=['train', 'validation'], data_dir=os.path.join(working_dir, "data/imagenet"), download=True, shuffle_files=True, as_supervised=True, with_info=True)
    	ds_val = ds_val.map(tf_norm_crop_resize_image, num_parallel_calls=AUTOTUNE)
	ds_train = ds_train.map(tf_norm_crop_resize_image, num_parallel_calls=AUTOTUNE)
    else:
	[ds_train, ds_val], ds_info = tfds.load('imagenet2012', split=['train', 'validation'], data_dir=os.path.join(working_dir, "data/imagenet"), download=True, shuffle_files=True, as_supervised=True, with_info=True)
    ds_val = ds_val.batch(batch_size)  
    ds_val = ds_val.prefetch(AUTOTUNE)
    ds_train = ds_train.map(tf_norm_crop_resize_image, num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(AUTOTUNE)
    return ds_train, ds_val





# Use this function to normalize, crop and resize images.
def tf_norm_crop_resize_image(image, label, resize_dim = (224, 224)):
    """Normalizes image to [0.,1.], crops to dims (64, 64, 3)
    and resizes to `resize_dim`, returning an image tensor."""
    image = tf.cast(image, tf.float32)/255.
    image = tf.image.resize_with_crop_or_pad(image, 112, 112)
    return image, tf.one_hot(indices=label, depth=1000)
    

