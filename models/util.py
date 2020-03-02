import tensorflow as tf
from ddsp import spectral_ops

def compute_mel(audio,
                sample_rate=16000,
                lo_hz=0.0,
                hi_hz=8000.0,
                bins=64,
                fft_size=2048,
                overlap=0.75,
                pad_end=True):
    '''Compute Mel spectrogram with Tensorflow.'''
    mag = spectral_ops.compute_mag(audio, fft_size, overlap, pad_end)
    num_spectrogram_bins = int(mag.shape[-1])
    linear_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        bins, num_spectrogram_bins, sample_rate, lo_hz, hi_hz)
    mel = tf.tensordot(mag, linear_to_mel_matrix, 1)
    mel.set_shape(mag.shape[:-1].concatenate(linear_to_mel_matrix.shape[-1:]))
    return mel