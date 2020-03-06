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
        with open(metadata_path, "rb") as metadata_file:
            metadata = pickle.load(metadata_file)
        audio_rate = metadata["audio_rate"]
        return audio, audio_rate

    def plot_mag_spectrum(audio, audio_rate, n_fft=2048, f_band=None):
        '''Plot magnitude spectrum using DDSP's compute_mag.'''
        mag = spectral_ops.compute_mag(audio, n_fft).numpy().T
        plt.figure()
        plt.imshow(mag, origin="lower")
        plt.show()

    def get_mel_spectrum(audio, audio_rate, n_fft=2048, n_mels=None, f_band=None):
        '''Plot magnitude spectrum using DDSP's compute_mag.'''
        n_mels = int(n_fft/16) if n_mels is None else n_mels
        if f_band is None:
            f_band = (0., audio_rate/2)
        mag = compute_mel(audio, sample_rate=audio_rate, lo_hz=f_band[0],
            hi_hz=f_band[1], bins=n_mels, fft_size=n_fft).numpy().T
        return mag
    
    def get_log_mel_spectrum(*args, **kwargs):
        mel = get_mel_spectrum(*args, **kwargs)
        return spectral_ops.safe_log(mel)
    
    def hz_to_mel(f):
        return 2595 * np.log10(1 + f/700)
    
    def mel_to_hz(m):
        return 700*(np.exp(m/1127) - 1)
    
    def df_dm(m):
        return 700/1127*(np.exp(m/1127))
    
    def get_closest_n_fft(fs, df, n_options):
        n_options = np.flip(np.sort(n_options.astype(np.int64)))
        n = fs/df
        N = len(n_options)
        for i in range(2, N):
            if n > n_options[i]:
                return [int(n_options[j]) for j in range(i, N)]
        return [int(n_options[-1])]
    
    get_spectrum = {"mag": plot_mag_spectrum,
                    "mel": get_mel_spectrum,
                    "logmel": get_log_mel_spectrum}[spec_type]

    audio, audio_rate = get_first_window_of_dataset()
    f_n = audio_rate/2
    f_max = f_n
    f_min = 0.0
    n_bands = 1
    #start = int(np.log2(128))
    #end = int(np.log2(16384))
    #num = end-start+1
    n_fft_options = 2**np.arange(0, 16)
    m_max = hz_to_mel(f_max)
    m_min = hz_to_mel(f_min)
    n_mels_max = int((m_max - m_min)/n_bands/4)
    m_all = np.linspace(m_min, m_max, n_bands+1)
    m_los = m_all[:-1]
    m_his = m_all[1:]
    f_los = mel_to_hz(m_los)
    f_his = mel_to_hz(m_his)
    #f_his = f_n/2**np.arange(11, -1, -1)
    #f_los = f_his/2
    #m_his = hz_to_mel(f_his)
    #m_los = hz_to_mel(f_los)
    d_m = (m_his - m_los)/n_mels_max
    d_fs = df_dm(m_los) * d_m
    for i, f_lo in enumerate(f_los):
        f_hi = f_his[i]
        d_f = d_fs[i]
        f_band = (f_lo, f_hi)
        for j, n_fft in enumerate(get_closest_n_fft(audio_rate, d_f, n_fft_options)):
            _, ax = plt.subplots(1, 1, figsize=(12,8))
            n_mels = int(n_mels_max/2**j)
            mag = get_spectrum(audio, audio_rate, n_fft=n_fft, n_mels=n_mels, f_band=f_band)
            plt.imshow(mag, origin="lower", cmap="magma")
            ax.set_aspect("auto")
            ax.set_title("n_fft: %d, n_mels: %d, f_band: (%.1f, %.1f)" % (n_fft, n_mels, f_lo, f_hi))
            plt.show()

def experiment_plot_mel_spectrum_given_time_freq_res(time_res=None, freq_res=None, spec_type="logmel"):
    '''
    Plot log Mel spectra with FFT sizes corresponding to given lower bounds on
    time and frequency resolution.

    My assumption at this point is that we need all information at all
    frequencies; splitting into frequency bands and only looking at the
    corresponding frequency resolution is not good if the model can
    generate time variations at a faster rate (read my discussion for
    experiment 200304_1_hpn_ford_mini_adaptive_mel for details).

    Author: Anton Lundberg
    Date:   2020-03-05
    '''
    def get_first_window_of_dataset(data_dir="./data/tfrecord/ford"):
        '''Get audio and sample rate of first window in a dataset.'''
        data_provider = TFRecordProvider(data_dir)
        data_iter = iter(data_provider.get_batch(1, shuffle=False))
        batch = next(data_iter)
        audio = batch["audio"].numpy()[0,:]
        return audio, data_provider.audio_rate, data_provider.input_rate

    def get_mel_spectrum(audio, audio_rate, n_fft=2048, n_mels=None, f_band=None, overlap=.75):
        '''Plot magnitude spectrum using DDSP's compute_mag.'''
        n_mels = int(n_fft/16) if n_mels is None else n_mels
        if f_band is None:
            f_band = (0., audio_rate/2)
        mag = compute_mel(audio, sample_rate=audio_rate,
                                 lo_hz=f_band[0],
                                 hi_hz=f_band[1],
                                 bins=n_mels,
                                 fft_size=n_fft,
                                 overlap=overlap).numpy().T
        return mag
    
    def get_log_mel_spectrum(*args, **kwargs):
        mel = get_mel_spectrum(*args, **kwargs)
        return spectral_ops.safe_log(mel)
    
    def hz_to_mel(f):
        return 2595 * np.log10(1 + f/700)
    
    def mel_to_hz(m):
        return 700*(np.exp(m/1127) - 1)
    
    def df_dm(m):
        return 700/1127*(np.exp(m/1127))
    
    def dm_df(f):
        return 1127/(700 + f)
    
    def get_n_fft_max(fs, df, n_fft_options=None):
        '''n_fft_options = np.flip(np.sort(n_fft_options.astype(np.int64)))
        n = fs/df
        if n > n_fft_options[0]:
            return n_fft_options[0]
        for i in range(len(n_fft_options)-1):
            if n < n_fft_options[i]:
                return n_fft_options[i]
        return n_fft_options[-1]'''
        return 2**int(np.log2(fs/df) + 1)
    
    def get_n_fft_min(fs, dt):
        return 2**int(np.log2(fs*dt))
    
    get_spectrum = {"mel": get_mel_spectrum,
                    "logmel": get_log_mel_spectrum}[spec_type]

    audio, audio_rate, input_rate = get_first_window_of_dataset()
    f_n = audio_rate/2
    if time_res is None:
        time_res = 1/input_rate
    if freq_res is None:
        freq_res = 2.0
    f_max = f_n
    f_min = 0.0
    m_max = hz_to_mel(f_max)
    m_min = hz_to_mel(f_min)
    n_fft_min = get_n_fft_min(audio_rate, time_res)
    n_fft_max = get_n_fft_max(audio_rate, freq_res)
    n_fft_list = 2**np.arange(np.log2(n_fft_min), np.log2(n_fft_max), dtype=np.int64)
    logging.info("Will use FFT sizes: %s" % str(n_fft_list))
    for n_fft in n_fft_list:
        df = audio_rate/n_fft
        dm = dm_df(f_min)*df
        n_mels = int((m_max - m_min)/dm)
        _, ax = plt.subplots(1, 1, figsize=(12,8))
        mag = get_spectrum(audio, audio_rate, n_fft=n_fft, n_mels=n_mels, overlap=3/4)
        plt.imshow(mag, origin="lower", cmap="magma")
        ax.set_aspect("auto")
        ax.set_title("n_fft: %d, n_mels: %d" % (n_fft, n_mels))
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
      - use time representation for latent f0
    
    Author: Anton Lundberg
    Date:   2020-02-28
    '''
    ckpt_dir = "./data/weights/200228_3_hpn_ford_mini_time_input"
    data_dir = "./data/tfrecord/ford_mini"
    experiment_ford_helper(ckpt_dir, data_dir, f0_denom=4., feature_domain="time")

def experiment_200228_4_hpn_ford_time_input():
    '''
    Generate Ford engine sounds from a harmonic-plus-noise model trained on to
    1 minute of recordings using a time representation into decoder instead of f0/4.

    Specifications, incl. differences from original DDSP autoencoder:
      - 1 minute of training sounds, first window of 1-min Ford dataset
      - log Mel spectrogram loss with (16384, 8192, ..., 128) FFT sizes
      - 2 equally sized frequency bands: (0-12k, 12k-24k) Hz 
      - divide f0 by 4 to take the 4 cylinders into account
      - use time representation for latent f0
    
    Author: Anton Lundberg
    Date:   2020-02-28
    '''
    ckpt_dir = "./data/weights/200228_4_hpn_ford_time_input"
    data_dir = "./data/tfrecord/ford"
    experiment_ford_helper(ckpt_dir, data_dir, f0_denom=4., feature_domain="time")

def experiment_200304_1_hpn_ford_mini_adaptive_mel():
    '''
    Generate Ford engine sounds from a harmonic-plus-noise model overfitted to
    4 seconds of recordings using an adaptive log Mel spectral loss.

    The idea was that, with a loss that divides the frequencies into equal-
    width bands in Mel space and includes one FFT size that prioritizes time
    and one that prioritizes frequency resolution, the training would converge
    to a better model that captures both the frequency characteristics and
    the transient envelopes.

    Specifications, incl. differences from original DDSP autoencoder:
      - 4 seconds of training sounds, first window of 1-min Ford dataset
      - adaptive log Mel spectral loss with 8 bands
      - divide f0 by 4 to take the 4 cylinders into account
      - use time representation for latent f0
    
    Author: Anton Lundberg
    Date:   2020-03-04
    '''
    ckpt_dir = "./data/weights/200304_1_hpn_ford_mini_adaptive_mel"
    data_dir = "./data/tfrecord/ford_mini"
    experiment_ford_helper(ckpt_dir, data_dir, f0_denom=4., feature_domain="time")

def experiment_200304_2_hpn_ford_500fps_adaptive_mel():
    '''
    Generate Ford engine sounds from a harmonic-plus-noise model overfitted to
    4 seconds of recordings using 500 frames per second (instead of 250).

    The idea was that with higher frame rate, we can get good enough resolution
    to capture the transitents.

    Specifications, incl. differences from original DDSP autoencoder:
      - full 1-min Ford dataset with f0 sampled at 500 fps
      - adaptive log Mel spectral loss with 8 bands
      - divide f0 by 4 to take the 4 cylinders into account
      - use time representation for latent f0
    
    Author: Anton Lundberg
    Date:   2020-03-04
    '''
    ckpt_dir = "./data/weights/200304_2_hpn_ford_500fps_adaptive_mel"
    data_dir = "./data/tfrecord/ford_500fps"
    experiment_ford_helper(ckpt_dir, data_dir, f0_denom=4., feature_domain="time")

def experiment_200305_1_hpn_ford_mini_freq_time_res_mel_loss():
    '''
    Generate Ford engine sounds from a harmonic-plus-noise model overfitted to
    4 seconds of recordings using the time-frequency resolution Mel loss.

    I learned from my mistakes that all time scales are important for all
    frequency bands. Here, I try a Mell spectral loss that adapts the number
    of Mel bins based on the definition of Mel frequency creates the list of
    FFT sizes with a lower bound based on the desired time resolution
    (preferrably given by the input frame rate) and an upper bound given by
    the desired frequency resolution.

    Specifications, incl. differences from original DDSP autoencoder:
      - first 4 seconds of 1-min Ford dataset
      - time-frequency resolution Mel loss
      - divide f0 by 4 to take the 4 cylinders into account
      - use time representation for latent f0
    
    Author: Anton Lundberg
    Date:   2020-03-05
    '''
    ckpt_dir = "./data/weights/200305_1_hpn_ford_mini_freq_time_res_mel_loss"
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
        '''mag_in = spectral_ops.compute_mag(audio_in, size=8192).numpy().T
        plt.imshow(mag_in, origin="lower")
        plt.show()
        pdb.set_trace()'''
        n_fft = 4096
        n_mels = int(n_fft/8)
        audio_dict = {"recording": audio_in, "synthesized": audio_out}
        for key in audio_dict.keys():
            plt.figure()
            plot_audio_f0(audio_dict[key], data_provider.audio_rate, f0, data_provider.input_rate, title=key, n_fft=n_fft, n_mels=n_mels)
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
    #experiment_plot_mag_spectrum("logmel")
    #experiment_200226_1_hpn_ford()
    #experiment_200227_1_hpn_ford_cyl()
    #experiment_200228_1_hpn_ford_mini_mel()
    #experiment_200228_2_hpn_ford_mini_hipass()
    #experiment_200228_3_hpn_ford_mini_time_input()
    #experiment_200228_4_hpn_ford_time_input()
    #experiment_200304_1_hpn_ford_mini_adaptive_mel()
    #experiment_200304_2_hpn_ford_500fps_adaptive_mel()
    #experiment_plot_mel_spectrum_given_time_freq_res()
    experiment_200305_1_hpn_ford_mini_freq_time_res_mel_loss()

if __name__ == "__main__":
    app.run(main)
