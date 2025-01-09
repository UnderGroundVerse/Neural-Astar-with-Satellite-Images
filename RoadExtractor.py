import keras as ks
from keras import layers
import numpy as np

class RoadExtractor:
    def __init__(self, input_shape=(256, 256, 3)):
        self.input_shape = input_shape


    def build_model(self):
        model = ks.models.Sequential([

        layers.Input(shape=self.input_shape),
        layers.Conv2D(32, (3, 3), strides=2 ,activation='relu', padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(64, (3, 3), strides=1 ,activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), strides=1 ,activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(128, (3, 3), strides=1 ,activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), strides=1 ,activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Dropout(0.5),
        layers.BatchNormalization(),



        layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), strides=1, activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=1, padding='same'),
        layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), strides=1 ,activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=1, padding='same'),
        layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same'),
        layers.Conv2D(32, (3, 3), strides=1 ,activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=1, padding='same'),
        layers.Conv2DTranspose(3, (3, 3), strides=2, activation='relu', padding='same'),
        layers.BatchNormalization(),

        layers.Conv2D(1, (3,3) ,strides=1 , padding='same' ,activation='sigmoid'),

])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    
    def save_model(model, name):
        model.save(f'trained_models/{name}.keras')
    def load_model(name):
        return ks.models.load_model(f'trained_models/{name}.keras')
    def get_predction(model, image , threshold=0.1):
        pred_image = model.predict(image)
        pred_image = pred_image[0]
        pred_image = np.where(pred_image > threshold, 1, 0)
        return pred_image
        
        
        
    