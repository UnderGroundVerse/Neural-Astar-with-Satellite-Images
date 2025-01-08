from tensorflow.keras import Model 
from tensorflow.keras.layers import Dense
from Transformer import Encoder ,Decoder


class Transformer(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        self.final_layer = Dense(target_vocab_size)

    def call(self, inputs, targets, training, look_ahead_mask, padding_mask):
        enc_output = self.encoder(inputs, training, padding_mask)
        dec_output = self.decoder(targets, enc_output, training, look_ahead_mask, padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output


    