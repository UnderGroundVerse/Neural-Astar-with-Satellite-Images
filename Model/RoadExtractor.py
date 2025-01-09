import keras as ks
from keras import layers

class RoadExtractor:
    def __init__(self, input_shape=(256, 256, 3)):
        self.input_shape = input_shape


    def build_model(self):
        model = ks.Sequential()
        model.add(layers.Input(shape=self.input_shape))
        #Encoder
        
        model.add(layers.Conv2D(32, (3, 3), strides=2 ,activation='relu', padding='same'))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2), padding='same'))
        model.add(layers.Conv2D(64, (3, 3),strides=1 , activation='relu', padding='same'))
        model.add(layers.Conv2D(64, (3, 3),strides=1 , activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2), padding='same'))
        model.add(layers.Conv2D(128, (3, 3), strides=1 ,activation='relu', padding='same'))
        model.add(layers.Conv2D(128, (3, 3), strides=1 ,activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2), padding='same'))
        model.add(layers.Dropout(0.5))
        model.add(layers.BatchNormalization())

        #Decoder
        model.add(layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same'))
        model.add(layers.Conv2D(128, (3, 3), strides=1, activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2), strides=1, padding='same'))
        model.add(layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same'))
        model.add(layers.Conv2D(64, (3, 3), strides=1 ,activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2), strides=1, padding='same'))
        model.add(layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same'))
        model.add(layers.Conv2D(32, (3, 3), strides=1 ,activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2), strides=1, padding='same'))
        model.add(layers.Conv2DTranspose(3, (3, 3), strides=2, activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(1, (3,3) ,strides=1 , padding='same' ,activation='sigmoid'))
        return model
        
    