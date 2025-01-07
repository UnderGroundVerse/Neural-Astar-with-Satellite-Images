
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import pickle


class Heuristic_model:
    def create_feed_forward_heuristic_model(input_dim):
            model = Sequential([
            Input(shape=(input_dim,)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')
        ])
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return model
        
    def save_model_as_pickle(model, name):  
        with open('trained_models/'+name+'.pkl', 'wb') as file:  
            pickle.dump(model, file)
            
    def load_model_from_pickle(name):
        with open('trained_models/'+name+'.pkl', 'rb') as file:  
            model = pickle.load(file)
        return model
        
        
    