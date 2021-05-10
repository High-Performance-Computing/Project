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


def _mask(model, dictionary):
        for layer in model.layers[:-1]:
                if layer.count_params() > 0:
                        mask_layer = dictionary.get(layer.name) # boolean array 
                        new_weights = tf.constant(mask_layer, dtype=tf.float32)*tf.constant(layer.get_weights(), dtype=tf.float32)
                        layer.set_weights(new_weights)
        return model

def apply_masking_layer(layer):
    print(f"LAYER COUNT PARAMS: {layer.count_params()}")
    if layer.count_params() > 0:
            #count += 1
            print(layer.name)
            #print('hey')
            mask_layer = dictionary.get(layer.name) # boolean array 
            new_weights = tf.constant(mask_layer, dtype=tf.float32)*tf.constant(layer.get_weights(), dtype=tf.float32)
            layer.set_weights(new_weights)

if __name__=='__main__':
                global dictionary
                global model_MNV2
                model_MNV2 = tf.keras.applications.MobileNetV2(input_shape=(112, 112, 3), alpha=1.0, include_top=True, weights=None, classes=1000, classifier_activation='softmax')
                dictionary = get_masking(model_MNV2)

                import multiprocessing
                from multiprocessing import set_start_method
                from multiprocessing import Pool
                #set_start_method("spawn")
                list_layers = [l for l in model_MNV2.layers[:10]]
                num_processes = multiprocessing.cpu_count()
                
                with Pool(processes=5) as pool:
                        pool.map(apply_masking_layer, list_layers)
