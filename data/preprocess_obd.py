'''
Script for synching OBD data with audio recording and storing results.
'''
import pickle
from absl import flags, app
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import correlate
import librosa
from librosa.display import specshow
from ddsp.spectral_ops import compute_f0, compute_loudness, _CREPE_FRAME_SIZE, _CREPE_SAMPLE_RATE
import sys
import pdb

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "./raw/obd/ford/1min", "Directory where data is located.")
flags.DEFINE_string("audio_filename", "1.wav", "File name of recorded audio.")
flags.DEFINE_list("commands", ["RPM", "SPEED", "THROTTLE_POS"], "Command corresponding to quantities to preprocess.")
flags.DEFINE_enum("engine_type", "four-stroke", ["four-stroke", "two-stroke"], "Type of engine (determines relation between f0 and RPM).")
flags.DEFINE_string("f0_path", None, "Path to f0 array, if it has been estimated previously.")
flags.DEFINE_list("plots", ["obd", "f0-rpm", "f0-audio"], "Which plots to show.")

def preprocess_obd_data(c_list):
    '''Preprocess OBD data.'''
    # Convert Quantity object to dimensionless numpy array.
    n = len(c_list.values)
    y = np.zeros(n)
    for i in range(n):
        try:
            y[i] = c_list.values[i].magnitude
        except:
            y[i] = np.nan
    
    # Convert absolute time in ns to relative time in s.
    t = np.array(c_list.times)
    t = t - t[0]

    # Remove NaN elements
    t = t[~np.isnan(y)]
    y = y[~np.isnan(y)]

    return y, t

def main(argv):
    # Load OBD data
    n_commands = len(FLAGS.commands)
    c_dict = dict()
    if "obd" in FLAGS.plots:
        _, axes = plt.subplots(n_commands, 1, figsize=(15, 3*n_commands))
    for i, c in enumerate(FLAGS.commands):
        c_pickle_path = os.path.join(FLAGS.data_dir, c + ".pickle")
        print("Unpickling %s data..." % c)
        c_list = pickle.load(open(c_pickle_path, "rb"))

        print("Preprocessing data...")
        unit = str(c_list.values[0].units)
        y, t = preprocess_obd_data(c_list)
        c_dict[c] = {"values": y, "times": t}

        if "obd" in FLAGS.plots:
            print("Plotting data...")
            ax = axes[i]
            ax.plot(t, y, "-x")
            ax.set_title(c)
            ax.set_xlabel("time [s]")
            ax.set_ylabel("%s" % unit)
    if "obd" in FLAGS.plots:
        plt.tight_layout()
    
    # Estimate f0 from audio (if not already done)
    f0_frame_size = _CREPE_FRAME_SIZE // 2#16
    f0_frame_rate = _CREPE_SAMPLE_RATE / f0_frame_size # frame rate of f0 and loudness features
    f0_path = os.path.join(FLAGS.data_dir, "f0.npy")
    audio_path = os.path.join(FLAGS.data_dir, FLAGS.audio_filename)
    audio, sample_rate = librosa.load(audio_path, None)
    if os.path.exists(f0_path):
        print("Using precomputed f0.")
        f0 = np.load(f0_path)
    else:
        print("Estimating f0 using CREPE...")
        audio_lores = librosa.core.resample(audio, sample_rate, _CREPE_SAMPLE_RATE)
        f0, _ = compute_f0(audio_lores, _CREPE_SAMPLE_RATE, f0_frame_rate)
        np.save(f0_path, f0)

    # Upsample RPM signal to match f0 signal
    print("Upsampling RPM signal...")
    f0_times = np.arange(0., len(f0)/f0_frame_rate, 1/f0_frame_rate)
    rpm = c_dict["RPM"]["values"]
    f0_rpm_lores = rpm/60.
    f0_rpm_lores_times = c_dict["RPM"]["times"]
    if FLAGS.engine_type == "four-stroke":
        f0_rpm_lores *= 2
    f0_rpm_times = f0_times[f0_times < np.max(f0_rpm_lores_times)]
    f0_rpm_func = interp1d(f0_rpm_lores_times, f0_rpm_lores, kind="cubic")
    f0_rpm = f0_rpm_func(f0_rpm_times)

    # Find lag between f0 and RPM
    print("Calculating lag between f0 and RPM...")
    xcorr = correlate(f0 - np.mean(f0), f0_rpm - np.mean(f0_rpm))
    lag = xcorr.argmax() - (len(f0_rpm) - 1)
    print("Found lag: %d frames (%.3f seconds)" % (lag, lag/f0_frame_rate))
    if "f0-rpm" in FLAGS.plots:
        _, axes = plt.subplots(2, 1, figsize=(15, 6))
        axes[0].plot(f0_times, f0, label="f0")
        axes[0].plot(f0_rpm_times, f0_rpm, label="f0-rpm")
        axes[0].set_title("Before alignment")
        axes[1].plot(f0_times - lag/f0_frame_rate, f0, label="f0")
        axes[1].plot(f0_rpm_times, f0_rpm, label="f0-rpm")
        axes[1].set_title("After alignment")
        plt.tight_layout()
    
    # Trim audio according to lag
    start = int(lag*sample_rate/f0_frame_rate)
    end = start + int(len(f0_rpm)*sample_rate/f0_frame_rate) + 1
    audio_trimmed = audio[start:end]
    print("Trimmed audio is %.3f seconds." % (len(audio_trimmed)/float(sample_rate)))
    if "f0-audio" in FLAGS.plots:
        fmax = 2**13
        n_fft = 2**13
        plt.figure(figsize=(15, 6))
        S = librosa.feature.melspectrogram(y=audio_trimmed, sr=sample_rate, n_fft=n_fft, n_mels=1024, fmax=fmax)
        S_dB = librosa.power_to_db(S, ref=np.max)
        ax = specshow(S_dB, x_axis='time', y_axis='mel', sr=sample_rate, fmax=fmax)
        f0_h, = ax.plot(f0_rpm_times, f0_rpm, "--")
        ax.set_ylim((0, 5*np.max(f0_rpm)))
        ax.set_xlabel("time [s]")
        ax.set_ylabel("frequency [Hz]")
        ax.legend([f0_h], ["f0-rpm"], loc="upper right")
        plt.tight_layout()
    
    plt.show()
    print("Done.")

if __name__ == "__main__":
    app.run(main)