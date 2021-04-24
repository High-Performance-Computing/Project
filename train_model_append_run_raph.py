from loading_imagenet import load_data
import os
import tensorflow_datasets as tfds
import keras
from keras import backend as K
from keras.metrics import categorical_crossentropy
from keras.models import Model
from keras.applications import MobileNet
import numpy as np
from keras.optimizers import Adam
import tensorflow as tf
import time
import wandb
from wandb.keras import WandbCallback
from keras.callbacks import ModelCheckpoint, LambdaCallback
import tempfile


# faire un callback en plus
# use the dictionary



def add_regularization(model, penalty):
	regularizer=tf.keras.regularizers.l2(penalty)
	for layer in model.layers:
        	for attr in ['kernel_regularizer']:
        		if hasattr(layer, attr):
        			setattr(layer, attr, regularizer)
	model_json = model.to_json()
	tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
	model.save_weights(tmp_weights_path)
	model = tf.keras.models.model_from_json(model_json)
	model.load_weights(tmp_weights_path, by_name=True)
	return model	

def train_model():
	batch_size = 512 
	ds_train, ds_val = load_data(batch_size)
	initial_learning_rate = 0.045
	epochs = 150
	devices = ['GPU:0', 'GPU:1', 'GPU:2', 'GPU:3']
	strategy = tf.distribute.MirroredStrategy(devices=devices)
	with strategy.scope():
		model_MNV2 = tf.keras.applications.MobileNetV2(input_shape=(112, 112, 3), alpha=1.0, include_top=True, weights=None, classes=1000, classifier_activation='softmax')
		lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=2500, decay_rate=0.98, staircase=True) 
		m = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
		# optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, rho=0.9, momentum=0.9, name='RMSprop')
		# optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
		optimizer = tf.keras.optimizers.Adam()
		model_MNV2 = add_regularization(model_MNV2, penalty=0.00004)
		model_MNV2.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy', m])
	wandb.init(project="LTH at scale")
	wandb.run.name = 'Figthing Overfitting 3: Hyperparameters from the initial paper'
	config = wandb.config
	model_callbacks = ModelCheckpoint(filepath = 'saved_models_/saved_MobileNet_{epoch}', save_weights_only = False)
	history = model_MNV2.fit(ds_train, validation_data=ds_val, epochs=epochs, verbose = 1, callbacks=[model_callbacks, WandbCallback()])
	return history



if __name__=='__main__':
	train_model()
