import keras as ks
from keras import layers
import numpy as np

class RoadExtractor(ks.Model):
    def __init__(self, input_shape=(256, 256, 3)):
        super(RoadExtractor, self).__init__()
        self.input_shape = input_shape
        self.rd_layers = [
       layers.Conv2D(32, (3, 3), strides=1 ,activation='relu', padding='same'),
       layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
       layers.MaxPooling2D((2, 2), padding='same'),
       layers.Conv2D(64, (3, 3), strides=1 ,activation='relu', padding='same'),
       layers.Conv2D(64, (3, 3), strides=1 ,activation='relu', padding='same'),
       layers.MaxPooling2D((2, 2), padding='same'),
       layers.Conv2D(128, (3, 3), strides=1 ,activation='relu', padding='same'),
       layers.Conv2D(128, (3, 3), strides=1 ,activation='relu', padding='same'),
       layers.Dropout(0.5),
       layers.BatchNormalization(),

       layers.Conv2D(256, (3, 3), strides=2 ,activation='relu', padding='same'),
       layers.Conv2D(256, (3, 3), strides=2 ,activation='relu', padding='same'),


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
       layers.Conv2D(3, (3, 3), strides=1 ,activation='relu', padding='same'),
       layers.BatchNormalization(),

       layers.Conv2D(1, (3,3) ,strides=1 , padding='same' ,activation='sigmoid'),

      ]
    def call(self,inputs):
      x = (self.rd_layers[0])(inputs)
      for layer in self.rd_layers[1:]:
        x = layer(x)
      return x

    def build_model(self, optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']):
        inputs = ks.Input(shape=self.input_shape)
        outputs = self.call(inputs)
        self.model = ks.Model(inputs=inputs,outputs=outputs)
        self.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    def save_model(self, name):
        self.save_weights(f'trained_models/{name}.h5')
    def load_model(self, name):
        self.load_weights(f'trained_models/{name}.h5') 
    def get_predction(self, image , threshold=0.1):
        pred_image = self.predict(image)
        pred_image = pred_image[0]
        pred_image = np.where(pred_image > threshold, 1, 0)
        return pred_image
