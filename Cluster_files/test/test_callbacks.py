import os
from keras.applications import MobileNet
import numpy as np
import tensorflow as tf
import time
from tensorflow import keras
import sys
import numpy as np
import multiprocessing
from multiprocessing import Pool
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
                        dictionary_masking[layer.name]=tf.constant(mask_layer, dtype=tf.float32)
        return dictionary_masking

def _mask():  # profile this code for the report: what is computationally demanding ? 
        for layer in model_MNV2.layers[:-1]:
                if layer.count_params() > 0:
                        mask_layer = dictionary.get(layer.name) # boolean array 
                        new_weights = mask_layer*tf.constant(layer.get_weights(), dtype=tf.float32)
                        layer.set_weights(new_weights)

def apply_masking_layer(layer):
        mask_layer = dictionary.get(layer.name) # boolean array 
        new_weights = tf.constant(mask_layer, dtype=tf.float32)*tf.constant(layer.get_weights(), dtype=tf.float32)
        layer.set_weights(new_weights)
        
if __name__=='__main__':
                global dictionary
                global model_MNV2
                model_MNV2 = tf.keras.applications.MobileNetV2(input_shape=(112, 112, 3), alpha=1.0, include_top=True, weights=None, classes=1000, classifier_activation='softmax')
                dictionary = get_masking(model_MNV2)

                #CODE PROFILING USING TIME
                start = time.time()
                _mask()
                print(f'Serial time: {time.time() - start:.4f}s')
                
                # PROFILING USING CPROFILE
                pr = cProfile.Profile()
                pr.enable()
                _mask()
                pr.disable()
                pr.print_stats(sort='time')

                # MULTIPROCESSING
                import multiprocessing
                from multiprocessing import Pool
                list_layers=[]
                for l in model_MNV2.layers[:120]:
                        if l.count_params() > 0:
                                list_layers.append(l)
                num_processes = multiprocessing.cpu_count()
                start = time.time()
                with Pool(processes=num_processes) as pool:
                        pool.map(apply_masking_layer, list_layers)
                print('Multiprocessing', time.time() - start)
