'''
Collection of scripts for simple experiments.
'''
import pdb

import os
import time
import pickle
from absl import app, logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from ddsp import spectral_ops

from model_builder import ModelBuilder
from data_provider import TFRecordProvider
from models.util import compute_mel
from data.util import plot_audio_f0

# ========== PLOT MAGNITUDE SPECTRUM ==========
def experiment_plot_mag_spectrum(spec_type):
    '''
    Plot a magnitude spectrum for a given frequency band.

    With this experiment, we can investigate if it would make sense to split
    the loss function into different frequency bands with different FFT sizes.

    Author: Anton Lundberg
    Date:   2020-02-27
    '''
    def get_first_window_of_dataset(data_dir="./data/tfrecord/ford"):
        '''Get audio and sample rate of first window in a dataset.'''
        data_provider = TFRecordProvider(data_dir)
        data_iter = iter(data_provider.get_batch(1, shuffle=False))
        batch = next(data_iter)
        audio = batch["audio"].numpy()[0,:]
        metadata_path = os.path.join(data_dir, "metadata.pickle")
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        audio_rate = metadata["audio_rate"]
        return audio, audio_rate

    def plot_mag_spectrum(audio, audio_rate, n_fft=2048, f_band=None):
        '''Plot magnitude spectrum using DDSP's compute_mag.'''
        mag = spectral_ops.compute_mag(audio, n_fft).numpy().T
        plt.figure()
        plt.imshow(mag, origin="lower")
        plt.show()

    def plot_mel_spectrum(audio, audio_rate, n_fft=2048, f_band=None):
        '''Plot magnitude spectrum using DDSP's compute_mag.'''
        n_mels = int(n_fft/16)
        if f_band is None:
            f_band = (0., audio_rate/2)
        mag = compute_mel(audio, sample_rate=audio_rate, lo_hz=f_band[0],
            hi_hz=f_band[1], bins=n_mels, fft_size=n_fft).numpy().T
        plt.imshow(mag, origin="lower")
    
    plot_spectrum = {"mag": plot_mag_spectrum, "mel": plot_mel_spectrum}[spec_type]

    audio, audio_rate = get_first_window_of_dataset()
    start = int(np.log2(128))
    end = int(np.log2(16384))
    num = end-start+1
    band_width = 5000
    d_f = (audio_rate/2 - band_width)/(num - 1)
    f_centers_start = 250
    f_centers_end = 18000
    f_centers = np.logspace(np.log10(f_centers_start), np.log10(f_centers_end), num=num)
    widths = np.logspace(np.log10(2*f_centers_start), np.log10(2*(audio_rate/2-f_centers_end)), num=num)
    f_n = audio_rate/2
    for i, n in enumerate(np.flip(np.logspace(start, end, num=num, base=2))):
        #f_band = (i*d_f, i*d_f+band_width)
        #f_band = (int(f_centers[i]-widths[i]/2+.5), int(f_centers[i]+widths[i]/2+.5))
        if i < num/4:
            f_band = (0, f_n/4)
        elif i < num/2:
            f_band = (f_n/4, f_n/2)
        elif i < 3*num/4:
            f_band = (f_n/2, 3*f_n/4)
        else:
            f_band = (3*f_n/4, f_n)
        _, ax = plt.subplots(1, 1, figsize=(12,8))
        plot_spectrum(audio, audio_rate, n_fft=int(n), f_band=f_band)
        ax.set_aspect("auto")
        plt.show()

# ========== GENERATE FORD ENGINE SOUNDS ==========
def experiment_200226_1_hpn_ford():
    '''
    Generate Ford engine sounds from a harmonic-plus-noise model trained on
    1 minute of sound.

    The model uses the default DDSP decoder.
    
    Author: Anton Lundberg
    Date:   2020-02-26
    '''
    ckpt_dir = "./data/weights/200226_1_hpn_ford"
    data_dir = "./data/tfrecord/ford"
    experiment_ford_helper(ckpt_dir, data_dir)

def experiment_200227_1_hpn_ford_cyl():
    '''
    Generate Ford engine sounds from a harmonic-plus-noise model trained on
    1 minute of sound.

    Specifications, incl. difference from DDSP decoder:
      - Divides the f0 signal by 4 to account for the 4 cylinders firing at
        uneven intensities
      - Uses a spectrogram loss FFT sizes (8192, 4096, ..., 64)
    
    Author: Anton Lundberg
    Date:   2020-02-27
    '''
    ckpt_dir = "./data/weights/200227_1_hpn_ford_cyl"
    data_dir = "./data/tfrecord/ford"
    experiment_ford_helper(ckpt_dir, data_dir, f0_denom=4.,
                                               n_harmonic_distribution=100,
                                               feature_domain="freq-old")

def experiment_200228_1_hpn_ford_mini_mel():
    '''
    Generate Ford engine sounds from a harmonic-plus-noise model overfitted to
    4 seconds of recordings.

    Specifications, incl. differences from original DDSP autoencoder:
      - 4 seconds of training sounds, first window of 1-min Ford dataset
      - log Mel spectrogram loss with (16384, 8192, ..., 128) FFT sizes
      - divide f0 by 4 to take the 4 cylinders into account
      - 200 parameters in filtered noise processor
    
    Author: Anton Lundberg
    Date:   2020-02-28
    '''
    ckpt_dir = "./data/weights/200228_1_hpn_ford_mini_mel"
    data_dir = "./data/tfrecord/ford_mini"
    experiment_ford_helper(ckpt_dir, data_dir, f0_denom=4.,
                                               n_noise_magnitudes=200,
                                               feature_domain="freq-old")

def experiment_200228_2_hpn_ford_mini_hipass():
    '''
    Generate Ford engine sounds from a harmonic-plus-noise model overfitted to
    4 seconds of recordings.

    Specifications, incl. differences from original DDSP autoencoder:
      - 4 seconds of training sounds, first window of 1-min Ford dataset
      - log Mel spectrogram loss with (16384, 8192, ..., 128) FFT sizes
      - divide f0 by 4 to take the 4 cylinders into account
      - 200 parameters in filtered noise processor
    
    Author: Anton Lundberg
    Date:   2020-02-28
    '''
    ckpt_dir = "./data/weights/200228_2_hpn_ford_mini_hipass"
    data_dir = "./data/tfrecord/ford_mini_hipass"
    experiment_ford_helper(ckpt_dir, data_dir, f0_denom=4.,
                                               n_noise_magnitudes=200,
                                               feature_domain="freq-old")

def experiment_200228_3_hpn_ford_mini_time_input():
    '''
    Generate Ford engine sounds from a harmonic-plus-noise model overfitted to
    4 seconds of recordings using a time representation into decoder instead of f0/4.

    Specifications, incl. differences from original DDSP autoencoder:
      - 4 seconds of training sounds, first window of 1-min Ford dataset
      - log Mel spectrogram loss with (16384, 8192, ..., 128) FFT sizes
      - 4 equally sized frequency bands: (0-6k, 6k-12k, 12k-18k, 18k-24k) Hz 
      - divide f0 by 4 to take the 4 cylinders into account
    
    Author: Anton Lundberg
    Date:   2020-02-28
    '''
    ckpt_dir = "./data/weights/200228_3_hpn_ford_mini_time_input"
    data_dir = "./data/tfrecord/ford_mini"
    experiment_ford_helper(ckpt_dir, data_dir, f0_denom=4., feature_domain="time")

def experiment_200228_4_hpn_ford_time_input():
    '''
    Generate Ford engine sounds from a harmonic-plus-noise model overfitted to
    4 seconds of recordings using a time representation into decoder instead of f0/4.

    Specifications, incl. differences from original DDSP autoencoder:
      - 4 seconds of training sounds, HIGH-PASS FILTERED first window of 1-min Ford dataset
      - log Mel spectrogram loss with (16384, 8192, ..., 128) FFT sizes
      - 2 equally sized frequency bands: (0-12k, 12k-24k) Hz 
      - divide f0 by 4 to take the 4 cylinders into account
    
    Author: Anton Lundberg
    Date:   2020-02-28
    '''
    ckpt_dir = "./data/weights/200228_4_hpn_ford_time_input"
    data_dir = "./data/tfrecord/ford"
    experiment_ford_helper(ckpt_dir, data_dir, f0_denom=4., feature_domain="time")

def experiment_ford_helper(ckpt_dir, data_dir, plot_type="spectrogram",
                                               sound_mode="save",
                                               f0_denom=1.,
                                               n_harmonic_distribution=60,
                                               n_noise_magnitudes=65,
                                               losses=None,
                                               feature_domain="freq"):
    '''
    Code general for all Ford experiments.
    '''
    logging.info("Loading data...")
    data_provider = TFRecordProvider(data_dir)
    input_tensor = data_provider.get_single_batch(batch_number=1)
    #input_tensor["f0"] = tf.convert_to_tensor(np.flip(np.arange(0., 100., 100./np.size(input_tensor["f0"]))), dtype=tf.float32)[tf.newaxis,:,tf.newaxis]
    #input_tensor["f0"] = tf.convert_to_tensor(np.arange(1., 200., 100./np.size(input_tensor["f0"])), dtype=tf.float32)[tf.newaxis,:,tf.newaxis]

    logging.info("Building model...")
    model = ModelBuilder(model_type="f0_rnn_fc_hpn_decoder",
                         audio_rate=data_provider.audio_rate,
                         input_rate=data_provider.input_rate,
                         window_secs=data_provider.example_secs,
                         f0_denom=f0_denom,
                         checkpoint_dir=ckpt_dir,
                         n_harmonic_distribution=n_harmonic_distribution,
                         n_noise_magnitudes=n_noise_magnitudes,
                         losses=losses,
                         feature_domain=feature_domain).build()

    logging.info("Normalizing inputs...")
    features = model.encode(input_tensor)

    logging.info("Synthesizing from f0 signal...")
    start = time.time()
    output_tensor = model.decode(features, training=False)
    time_elapsed = time.time() - start
    logging.info("Synthesis took %.3f seconds." % time_elapsed)

    logging.info("Plotting signals...")
    audio_in = features["audio"].numpy()[0,:]
    audio_out = output_tensor.numpy()[0,:]
    f0 = input_tensor["f0"].numpy()[0,:]
    f0_scaled = features["f0_scaled"].numpy()[0,:]
    if plot_type == "signal":
        _, ax = plt.subplots(4, 1, figsize=(10, 8))
        ax[0].plot(audio_in)
        ax[1].plot(audio_out)
        ax[2].plot(f0)
        ax[3].plot(f0_scaled)
    elif plot_type == "spectrogram":
        plt.figure()
        '''mag_in = spectral_ops.compute_mag(audio_in, size=8192).numpy().T
        plt.imshow(mag_in, origin="lower")
        plt.show()
        pdb.set_trace()'''
        n_fft = 8192
        plot_audio_f0(audio_in, data_provider.audio_rate, f0, data_provider.input_rate, title="recording", n_fft=n_fft)
        plt.figure()
        #mag_out = spectral_ops.compute_mag(audio_out, size=8192)
        plot_audio_f0(audio_out, data_provider.audio_rate, f0, data_provider.input_rate, title="synthesized", n_fft=n_fft)
    plt.show()

    if sound_mode == "play":
        logging.info("Playing original audio...")
        sd.play(audio_in, data_provider.audio_rate)
        sd.wait()
        logging.info("Playing synthesized audio...")
        sd.play(audio_out, data_provider.audio_rate)
        sd.wait()
    elif sound_mode == "save":
        audio_in_path = "./audio_in.wav"
        audio_out_path = "./audio_out.wav"
        logging.info("Saving recorded audio to '%s'..." % audio_in_path)
        sf.write(audio_in_path, audio_in, data_provider.audio_rate)
        logging.info("Saving synthesized audio to '%s'..." % audio_out_path)
        sf.write(audio_out_path, audio_out, data_provider.audio_rate)

def main(argv):
    #experiment_200226_1_hpn_ford()
    #experiment_200227_1_hpn_ford_cyl()
    #experiment_200228_1_hpn_ford_mini_mel()
    #experiment_200228_2_hpn_ford_mini_hipass()
    #experiment_200228_3_hpn_ford_mini_time_input()
    experiment_200228_4_hpn_ford_time_input()

if __name__ == "__main__":
    app.run(main)
