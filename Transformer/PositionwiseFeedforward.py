from tensorflow.keras.layers import Dense
import tensorflow as tf

class PositionwiseFeedforward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):
        super(PositionwiseFeedforward, self).__init__()
        self.d_model = d_model
        self.dff = dff
        self.dense1 = Dense(dff, activation='relu')
        self.dense2 = Dense(d_model)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x
    
if __name__ == '__main__':
    ff = PositionwiseFeedforward(512, 2048)
    x = tf.random.uniform((64, 50, 512))
    output = ff(x)
    print(output.shape)
    print(output)