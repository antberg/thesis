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
                 n_rnn=1,
                 ch=512,
                 layers_per_stack=3,
                 output_splits=(("amps", 1), ("harmonic_distribution", 40)),
                 name="f0_rnn_fc_decoder"):
        super().__init__(output_splits=output_splits, name=name)

        # Create layers.
        stack = lambda: nn.fc_stack(ch, layers_per_stack)
        self.f0_stack = stack()
        self.n_rnn = n_rnn
        self.rnn = [nn.rnn(rnn_channels, rnn_type)]
        for _ in range(self.n_rnn-1):
            self.rnn.append(nn.rnn(rnn_channels, rnn_type))
        self.out_stack = stack()
        self.dense_out = nn.dense(self.n_out)

    def decode(self, conditioning):
        f = conditioning["f0_scaled"]

        # Initial processing.
        f = self.f0_stack(f)

        # Run an RNN over the latents.
        x = self.rnn[0](f)
        for i in range(self.n_rnn-1):
            x = self.rnn[i+1](x)
        x = tf.concat([f, x], axis=-1)

        # Final processing.
        x = self.out_stack(x)
        return self.dense_out(x)

class MultiInputRnnFcDecoder(Decoder):
    '''
    RNN-FC decoder taking multiple inputs and outputting synthesizer parameters.
    Intended for training on a single sound source (one car model, e.g.).
    '''
    def __init__(self,
                 rnn_channels=512,
                 rnn_type="gru",
                 n_rnn=1,
                 ch=512,
                 layers_per_stack=3,
                 input_keys=["f0_scaled", "osc_scaled"],
                 output_splits=(("amps", 1), ("harmonic_distribution", 40)),
                 name="multi_input_rnn_fc_decoder"):
        super().__init__(output_splits=output_splits, name=name)
        self.input_keys = input_keys
        stack = lambda: nn.fc_stack(ch, layers_per_stack)

        # Layers.
        self.stacks = []
        for _ in range(self.n_in):
            self.stacks.append(stack())
        self.n_rnn = n_rnn
        self.rnn = [nn.rnn(rnn_channels, rnn_type)]
        for _ in range(self.n_rnn-1):
            self.rnn.append(nn.rnn(rnn_channels, rnn_type))
        self.out_stack = stack()
        self.dense_out = nn.dense(self.n_out)

    @property
    def n_in(self):
        return len(self.input_keys)

    def decode(self, conditioning):
        c = []
        for i in range(self.n_in):
            c.append(conditioning[self.input_keys[i]])

        # Initial processing.
        for i in range(self.n_in):
            c[i] = self.stacks[i](c[i])

        # Run an RNN over the latents.
        x = tf.concat(c, axis=-1)
        x = self.rnn[0](x)
        for i in range(self.n_rnn-1):
            x = self.rnn[i+1](x)
        x = tf.concat(c + [x], axis=-1)

        # Final processing.
        x = self.out_stack(x)
        return self.dense_out(x)