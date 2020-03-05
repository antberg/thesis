'''
Loss functions for comparing audio signals.
'''
import functools
from absl import logging
import tensorflow as tf
import numpy as np
from ddsp import spectral_ops
from ddsp.losses import mean_difference

from .util import compute_mel

tfkl = tf.keras.layers

# ================ HELPER FUNCTIONS ================

def hz_to_mel(f):
    '''Convert from Hz to Mel frequency.'''
    return 2595 * np.log10(1 + f/700)

def mel_to_hz(m):
    '''Convert from Mel frequency to Hz.'''
    return 700*(np.exp(m/1127) - 1)

def df_dm(m):
    '''Derivative of frequency [Hz] with respect to Mel frequency.'''
    return 700/1127*(np.exp(m/1127))

def dm_df(f):
    '''Derivative of Mel frequency with respect to frequency [Hz].'''
    return 1127/(700 + f)

# ================ LOSSES ================

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
        for i, loss_op in enumerate(loss_ops):
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

class AdaptiveMelSpectralLoss(tfkl.Layer):
    '''
    Multi-band, multi-scale Mel-spectrogram loss that divides frequencies into
    equal-width bands in Mel space and adapts FFT sizes these bands.
    '''
    N_FFT_OPTIONS = 2**np.arange(0, 16)
    
    def __init__(self,
                 sample_rate=16000,
                 n_bands=8,
                 loss_type="L1",
                 mag_weight=0.0,
                 logmag_weight=1.0,
                 name="adaptive_mel_spectral_loss"):
        super().__init__(name=name)
        self.sample_rate = sample_rate
        self.n_bands = n_bands
        self.loss_type = loss_type
        self.mag_weight = mag_weight
        self.logmag_weight = logmag_weight

    def call(self, audio, target_audio):
        loss = 0.0
        loss_ops = []

        f_n = self.sample_rate/2
        f_max = f_n
        f_min = 0.0
        m_max = self._hz_to_mel(f_max)
        m_min = self._hz_to_mel(f_min)
        n_mels_max = int((m_max - m_min)/self.n_bands/4)
        m_all = np.linspace(m_min, m_max, self.n_bands+1)
        m_los = m_all[:-1]
        m_his = m_all[1:]
        f_los = self._mel_to_hz(m_los)
        f_his = self._mel_to_hz(m_his)
        d_m = (m_his - m_los)/n_mels_max
        d_fs = self._df_dm(m_los) * d_m
        for i, f_lo in enumerate(f_los):
            f_hi = f_his[i]
            d_f = d_fs[i]
            for j, n_fft in enumerate(self._get_closest_n_fft(self.sample_rate, d_f, self.N_FFT_OPTIONS)):
                n_mels = int(n_mels_max/2**(2*j))
                loss_op = functools.partial(compute_mel,
                                            sample_rate=self.sample_rate,
                                            lo_hz=f_lo,
                                            hi_hz=f_hi,
                                            bins=n_mels,
                                            fft_size=n_fft)
                loss_ops.append(loss_op)

        # Compute loss for each fft size.
        for i, loss_op in enumerate(loss_ops):
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
    
    @staticmethod
    def _hz_to_mel(f):
        return 2595 * np.log10(1 + f/700)
    
    @staticmethod
    def _mel_to_hz(m):
        return 700*(np.exp(m/1127) - 1)
    
    @staticmethod
    def _df_dm(m):
        return 700/1127*(np.exp(m/1127))
    
    @staticmethod
    def _get_closest_n_fft(fs, df, n_options):
        n_options = np.flip(np.sort(n_options))
        n = fs/df
        for i in range(2, len(n_options)-1):
            if n > n_options[i]:
                return [int(n_options[i-2]), int(n_options[i])]
        return [int(n_options[-1])]

class TimeFreqResMelSpectralLoss(tfkl.Layer):
    '''
    Mel spectrogram loss that uses FFT sizes depending on given frequency
    and time resolutions.
    '''
    def __init__(self,
                 sample_rate=16000,
                 time_res=None,
                 freq_res=None,
                 loss_type="L1",
                 mag_weight=0.0,
                 logmag_weight=1.0,
                 name="time_freq_res_mel_spectral_loss"):
        if time_res is None:
            ValueError("Desired time resolution (time_res) must be set.")
        super().__init__(name=name)
        self.sample_rate = sample_rate
        self.loss_type = loss_type
        self.mag_weight = mag_weight
        self.logmag_weight = logmag_weight
        self.time_res = time_res
        self.freq_res = 2.0 if freq_res is None else freq_res
        self.loss_ops = []
        f_n = self.sample_rate/2
        f_max = f_n
        f_min = 0.0
        m_max = hz_to_mel(f_max)
        m_min = hz_to_mel(f_min)
        n_fft_min = self._get_n_fft_min()
        n_fft_max = self._get_n_fft_max()
        n_fft_list = 2**np.arange(np.log2(n_fft_min), np.log2(n_fft_max), dtype=np.int64)
        logging.info("Will use FFT sizes: %s" % str(n_fft_list))
        for n_fft in n_fft_list:
            df = self.sample_rate/n_fft
            dm = dm_df(f_min)*df
            n_mels = int((m_max - m_min)/dm)
            loss_op = functools.partial(compute_mel,
                                        sample_rate=self.sample_rate,
                                        lo_hz=f_min,
                                        hi_hz=f_max,
                                        bins=n_mels,
                                        fft_size=n_fft)
            self.loss_ops.append(loss_op)

    def call(self, audio, target_audio):
        loss = 0.0

        # Compute loss for each fft size.
        for loss_op in self.loss_ops:
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
    
    def _get_n_fft_max(self):
        return 2**int(np.log2(self.sample_rate/self.freq_res) + 1)
    
    def _get_n_fft_min(self):
        return 2**int(np.log2(self.sample_rate*self.time_res))