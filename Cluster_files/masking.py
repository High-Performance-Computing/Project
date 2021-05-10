import os
from keras.applications import MobileNet
import numpy as np
from keras.optimizers import Adam
import tensorflow as tf
import time
import wandb
from wandb.keras import WandbCallback
from keras.callbacks import ModelCheckpoint, LambdaCallback
import tempfile
from loading_imagenet import load_data
from tensorflow import keras
import sys
import numpy as np

global spark  # boolean variable in order to indicate whether or not you want to use Spark offline preprocessing
spark = True
global threshold
threshold = float(sys.argv[2])
global treillis_late_resetting
treillis_late_resetting = [5, 10, 15, 20, 25, 30, 35, 40]

def get_masking(model):
	dictionary_masking={}
	for layer in model.layers[:-1]:
		if layer.count_params() > 0:
			weights = layer.get_weights()
			mask_layer = []
			for weight in weights:
				if len(weight)>0:
					mask_layer.append(1*(np.abs(weight)>m))
			dictionary_masking[layer.name]=mask_layer
	return dictionary_masking


def _mask(model, dictionary):
	for layer in model.layers[:-1]:
		if layer.count_params() > 0:
			mask_layer = dictionary.get(layer.name) # boolean array 
			new_weights = tf.constant(mask_layer, dtype=tf.float32)*tf.constant(layer.get_weights(), dtype=tf.float32)
			layer.set_weights(new_weights)
	return model
			

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


def on_step_end(epoch, logs, sparse_MNV2):
        sparse_MNV2 = _mask(sparse_MNV2, dictionary)


def train_model(epoch, threshold):
	batch_size = 512
	ds_train, ds_val = load_data(batch_size, spark)
	initial_learning_rate = 0.045
	epochs = 150
	devices = ['GPU:0', 'GPU:1', 'GPU:2', 'GPU:3']
	strategy = tf.distribute.MirroredStrategy(devices=devices)
	with strategy.scope():
		model_MNV2 = tf.keras.models.load_model('saved_models_/saved_MobileNet_' + str(epoch))
		global dictionary
		dictionary = get_masking(model_MNV2) # this is the mask for all of the late resettings we'll do for this specific subarchitecture
		global sparse_MNV2
		sparse_MNV2 = _mask(model_MNV2, dictionary)
		lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=2500, decay_rate=0.98, staircase=True)
		m = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
		optimizer = tf.keras.optimizers.Adam()
		sparse_MNV2 = add_regularization(sparse_MNV2, penalty=0.00004)
		sparse_MNV2.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy', m])
		callback_sparsify = LambdaCallback(on_batch_end= lambda epoch, logs: on_step_end(epoch, logs, sparse_MNV2))
	wandb.init(project="LTH at scale")
	wandb.run.name = 'Masking Sparsification: threshold ' + str(threshold) + ' late resetting ' + str(epoch)
	config = wandb.config
	history = sparse_MNV2.fit(ds_train, validation_data=ds_val, epochs=epochs, verbose = 1, callbacks=[callback_sparsify, WandbCallback()])
	sparse_MNV2.save('masked_saved_models_/saved_MobileNet_t{threshold}_e{epoch}')
	return history


if __name__=='__main__':
	for epoch in treillis_late_resetting:
		train_model(epoch, threshold)
