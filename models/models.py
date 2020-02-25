'''Full models for neural audio synthesis.'''
from ddsp.training.models import Autoencoder
from ddsp.losses import SpectralLoss
from .preprocessors import F0Preprocessor
from .decoders import F0RnnFcDecoder
from .processors import HarmonicPlusNoise

import pdb

class F0RnnFcHPNDecoder(Autoencoder):
    '''Full RNN-FC decoder harmonic-plus-noise synthesizer stack for decoding f0 signals into audio.'''
    def __init__(self, window_secs=None, audio_rate=None, input_rate=None, name="f0-rnn-fc-hps-decoder"):
        # Initialize preprocessor
        preprocessor = F0Preprocessor()

        # Initialize decoder
        decoder = F0RnnFcDecoder(rnn_channels = 512,
                                 rnn_type = "gru",
                                 ch = 512,
                                 layers_per_stack = 3,
                                 output_splits = (('amps', 1),
                                                  ('harmonic_distribution', 60),
                                                  ('noise_magnitudes', 65)))
        
        # Initialize processor group
        processor_group = HarmonicPlusNoise(window_secs=window_secs,
                                            audio_rate=audio_rate,
                                            input_rate=input_rate)
        
        # Initialize losses
        losses = [SpectralLoss(loss_type="L1",
                               mag_weight=1.0,
                               logmag_weight=1.0)]
        
        # Call parent constructor
        super(F0RnnFcHPNDecoder, self).__init__(preprocessor=preprocessor,
                                                encoder=None,
                                                decoder=decoder,
                                                processor_group=processor_group,
                                                losses=losses)