
import tensorflow.python.keras as tf_keras
from keras import __version__
tf_keras.__version__ = __version__

import numpy as np
import tensorflow as tf

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Flatten, Input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


import pickle

class HeuristicModel:
    def create_feed_forward_heuristic_model(input_dim):
            model = Sequential([
            Input(shape=(input_dim,)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')
        ])
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return model
        
    def save_model(model, name):  
        model.save(f'trained_models/{name}.keras')
            
    def load_model(name):
        return load_model(f'trained_models/{name}.keras')
        
        
    
