'''
Loss functions for comparing audio signals.
'''
import functools
import tensorflow as tf
import numpy as np
from ddsp import spectral_ops
from ddsp.losses import mean_difference

from .util import compute_mel

tfkl = tf.keras.layers

class MelSpectralLoss(tfkl.Layer):
    '''Multi-band, multi-scale Mel-spectrogram loss.'''
    def __init__(self,
                 fft_sizes=(16384, 8192, 4096, 2048, 1024, 512, 256, 128),
                 sample_rate=16000,
                 n_bands=4,
                 loss_type="L1",
                 mag_weight=0.0,
                 logmag_weight=1.0,
                 name="mel_spectral_loss"):
        if np.log2(n_bands) % 1.0 != 0.0:
            raise ValueError("n_bands must be a power of 2.")
        if np.log2(len(fft_sizes)) % 1.0 != 0.0:
            raise ValueError("Number of layers must be a power of 2.")
        if n_bands > len(fft_sizes):
            raise ValueError("n_bands cannot exceed number of layers.")
        super().__init__(name=name)
        self.fft_sizes = fft_sizes
        self.sample_rate = sample_rate
        self.n_bands = n_bands
        self.loss_type = loss_type
        self.mag_weight = mag_weight
        self.logmag_weight = logmag_weight

    def call(self, audio, target_audio):
        loss = 0.0
        loss_ops = []

        n_layers = len(self.fft_sizes)
        f_n = self.sample_rate/2
        f_bands_ids = np.arange(0, self.n_bands).repeat(n_layers/self.n_bands)
        band_width = f_n/self.n_bands
        for i, n_fft in enumerate(self.fft_sizes):
            n_mels = int(n_fft/16) # TODO: this is ad-hoc; change for something more motivated
            f_lo = f_bands_ids[i]*band_width
            f_hi = f_lo + band_width
            loss_op = functools.partial(compute_mel,
                                        sample_rate=self.sample_rate,
                                        lo_hz=f_lo,
                                        hi_hz=f_hi,
                                        bins=n_mels,
                                        fft_size=n_fft)
            loss_ops.append(loss_op)

        # Compute loss for each fft size.
        for loss_op in loss_ops:
            target_mag = loss_op(target_audio)
            value_mag = loss_op(audio)

            # Add magnitude loss.
            if self.mag_weight > 0:
                loss += self.mag_weight * mean_difference(target_mag,
                                                        value_mag,
                                                        self.loss_type)

            # Add logmagnitude loss, reusing spectrogram.
            if self.logmag_weight > 0:
                target = spectral_ops.safe_log(target_mag)
                value = spectral_ops.safe_log(value_mag)
                loss += self.logmag_weight * mean_difference(target,
                                                            value,
                                                            self.loss_type)

        return loss