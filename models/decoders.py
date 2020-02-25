'''
Classes for decoder models that generate synthesizer parameters given some inputs.
'''
import tensorflow as tf
from ddsp.training.decoders import Decoder
from ddsp.training import nn

class F0RnnFcDecoder(Decoder):
    '''
    RNN-FC decoder taking f0 as input and outputting synthesizer parameters.
    Intended for training on a single sound source (one car model, e.g.).
    '''
    def __init__(self,
                 rnn_channels=512,
                 rnn_type="gru",
                 ch=512,
                 layers_per_stack=3,
                 output_splits=(("amps", 1), ("harmonic_distribution", 40)),
                 name="rnn_fc_decoder"):
        super().__init__(output_splits=output_splits, name=name)

        # Create layers.
        stack = lambda: nn.fc_stack(ch, layers_per_stack)
        self.f0_stack = stack()
        self.rnn = nn.rnn(rnn_channels, rnn_type)
        self.out_stack = stack()
        self.dense_out = nn.dense(self.n_out)

    def decode(self, conditioning):
        f = conditioning["f0_scaled"]

        # Initial processing.
        f = self.f0_stack(f)

        # Run an RNN over the latents.
        x = self.rnn(f)
        x = tf.concat([f, x], axis=-1)

        # Final processing.
        x = self.out_stack(x)
        return self.dense_out(x)