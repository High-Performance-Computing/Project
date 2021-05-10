import os
from keras.applications import MobileNet
import numpy as np
from keras.optimizers import Adam
import tensorflow as tf
import time
from keras.callbacks import ModelCheckpoint, LambdaCallback
import tempfile
#from loading_imagenet import load_data
from tensorflow import keras
import sys
import numpy as np
import multiprocessing
from multiprocessing import Pool
import time
from tqdm import tqdm
import cProfile

m = .5

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


def _mask():  # profile this code for the report: what is computationally demanding ? 
        for layer in model_MNV2.layers[:-1]:
                if layer.count_params() > 0:
                        mask_layer = dictionary.get(layer.name) # boolean array 
                        new_weights = tf.constant(mask_layer, dtype=tf.float32)*tf.constant(layer.get_weights(), dtype=tf.float32)
                        layer.set_weights(new_weights)

def apply_masking_layer(layer):
	#print(f"LAYER COUNT PARAMS: {layer.count_params()}")
	#print(layer.name)
	#if layer.name != 'global_average_pooling2d_2' and layer.name != 'out_relu':
	#if layer.count_params() > 0:
	mask_layer = dictionary.get(layer.name) # boolean array 
	new_weights = tf.constant(mask_layer, dtype=tf.float32)*tf.constant(layer.get_weights(), dtype=tf.float32)
	layer.set_weights(new_weights)	

if __name__=='__main__':
		global dictionary
		global model_MNV2
		model_MNV2 = tf.keras.applications.MobileNetV2(input_shape=(112, 112, 3), alpha=1.0, include_top=True, weights=None, classes=1000, classifier_activation='softmax')
		dictionary = get_masking(model_MNV2)
		#threads = 4
		start = time.time()
		_mask()
		print('Serial', time.time() - start)
		import cProfile

		pr = cProfile.Profile()
		pr.enable()

		_mask()

		pr.disable()
		pr.print_stats(sort='time')
		#cProfile.run('_mask()')
		#start = time.time()
		#with Pool(threads) as p:
        	#	tqdm(p.imap(apply_masking_layer, model_MNV2), total=len(model_MNV2.layers))
		#end = time.time()
		#print('Multiprocessing', end - start)
		import multiprocessing
		#from multiprocessing import set_start_method
		from multiprocessing import Pool
		#set_start_method("fork")
		#list_layers = [l for l in model_MNV2.layers[:-1]]
		list_layers=[]
		for l in model_MNV2.layers[:120]:
    			if l.count_params() > 0 and l.name != 'Conv_1_bn' and l.name != 'Conv_1':
        			list_layers.append(l)
		num_processes = multiprocessing.cpu_count()
		start = time.time()
		with Pool(processes=num_processes) as pool:
			pool.map(apply_masking_layer, list_layers)
		print('Multiprocessing', time.time() - start)
		#pool = mp.Pool(num_processes)
		#layers_ex = pool.map(apply_masking_layer, list_layers)
