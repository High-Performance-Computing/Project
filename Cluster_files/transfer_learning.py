import os
import tensorflow_datasets as tfds
from keras import backend as K
from keras.metrics import categorical_crossentropy
from keras.models import Model
import numpy as np
from keras.optimizers import Adam
import tensorflow as tf
import time
import wandb
from wandb.keras import WandbCallback
from train_model import add_regularization
from keras.callbacks import ModelCheckpoint, LambdaCallback


AUTOTUNE = tf.data.AUTOTUNE
location_winning_ticket = 'saved_models_/winning_ticket'

def load_cifar_100():
    ######### load the data ##########
    ds_train, ds_val = tfds.load(name="cifar100", split=['train', 'test'], as_supervised=True) 
    ds_val = ds_val.map(tf_norm_crop_resize_image, num_parallel_calls=AUTOTUNE)
    ds_val = ds_val.batch(128)  
    ds_val = ds_val.prefetch(AUTOTUNE)
    ds_train = ds_train.map(tf_norm_crop_resize_image, num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(AUTOTUNE)
    return ds_train, ds_val



# Use this function to normalize, crop and resize images.
def tf_norm_crop_resize_image(image, label, resize_dim = (224, 224)):
	"""Normalizes image to [0.,1.], crops to dims (64, 64, 3)
	and resizes to `resize_dim`, returning an image tensor."""
	image = tf.cast(image, tf.float32)/255.
	image = tf.image.resize_with_crop_or_pad(image, 112, 112)
	return image, tf.one_hot(indices=label, depth=1000)	



def finetune_winning_ticket(filepath=location_winning_ticket):
    ds_train, ds_val = load_cifar_100()
    initial_learning_rate = 5e-5 # recommended by the paper one ticket to win them all 
    epochs = 10
    devices = ['GPU:0', 'GPU:1', 'GPU:2', 'GPU:3']
    strategy = tf.distribute.MirroredStrategy(devices=devices)
    with strategy.scope():
        winning_ticket = tf.keras.models.load_model(filepath)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=2500, decay_rate=0.98, staircase=True)
        m = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
        optimizer = tf.keras.optimizers.Adam()
        winning_ticket = add_regularization(winning_ticket, penalty=0.00004)
        winning_ticket.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy', m])
    wandb.init(project="LTH at scale")
    wandb.run.name = 'Figthing Overfitting 3: Hyperparameters from the initial paper'
    config = wandb.config
    model_callbacks = ModelCheckpoint(filepath = 'saved_models_/saved_MobileNet_{epoch}', save_weights_only = False)
    history = winning_ticket.fit(ds_train, validation_data=ds_val, epochs=epochs, verbose = 1, callbacks=[model_callbacks, WandbCallback()])
    return history	



if __name__=='__main__':
	finetune_winning_ticket(filepath= 'saved_models_/winning_ticket')
