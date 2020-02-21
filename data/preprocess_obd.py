'''
Script for synching OBD data with audio recording and storing results.
'''
import pickle
from absl import flags, app, logging
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import correlate
import librosa
from librosa.display import specshow
from ddsp.spectral_ops import compute_f0, compute_loudness, _CREPE_FRAME_SIZE, _CREPE_SAMPLE_RATE
import pdb

from constants import DEFAULT_SAMPLE_RATE, DEFAULT_FRAME_RATE
from util import plot_data_dict, get_timestamp

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "./raw/obd/ford/1min", "Directory where OBD data and audio is located.")
flags.DEFINE_string("audio_filename", "1.wav", "File name of recorded audio.")
flags.DEFINE_string("save_dir", "./processed/obd/", "Directory where processed data will be stored.")
flags.DEFINE_string("data_name", get_timestamp(), "Name of data (will be used as filename when saving).")
flags.DEFINE_integer("sample_rate", DEFAULT_SAMPLE_RATE, "Sample rate of audio.")
flags.DEFINE_integer("frame_rate", DEFAULT_FRAME_RATE, "Sample rate of audio.")
flags.DEFINE_list("commands", ["RPM", "SPEED", "THROTTLE_POS"], "Command corresponding to quantities to preprocess.")
flags.DEFINE_string("interp_method", "linear", "Method for resampling OBD signals.")
flags.DEFINE_enum("engine_type", "four-stroke", ["four-stroke", "two-stroke"], "Type of engine (determines relation between f0 and RPM).")
flags.DEFINE_string("f0_path", None, "Path to f0 array, if it has been estimated previously.")
flags.DEFINE_list("plots", ["obd", "f0-rpm", "f0-audio", "data"], "Which plots to show.")
flags.DEFINE_string("plot_mode", "save", "Whether to save or show plots.")

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
    # Check preconditions
    if "RPM" not in FLAGS.commands:
        raise Exception("RPM must be part of the OBD commands since it is used for OBD-audio alignment.")
    
    # Create save folder
    save_dir = os.path.join(FLAGS.save_dir, FLAGS.data_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if FLAGS.plot_mode == "save":
        plots_dir = os.path.join(save_dir, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

    # Load OBD data
    n_commands = len(FLAGS.commands)
    c_dict = dict()
    if "obd" in FLAGS.plots:
        _, axes = plt.subplots(n_commands, 1, figsize=(15, 3*n_commands))
    for i, c in enumerate(FLAGS.commands):
        c_pickle_path = os.path.join(FLAGS.data_dir, c + ".pickle")
        logging.info("Unpickling %s data..." % c)
        c_list = pickle.load(open(c_pickle_path, "rb"))

        logging.info("Preprocessing data...")
        unit = str(c_list.values[0].units)
        y, t = preprocess_obd_data(c_list)
        c_dict[c] = {"values": y, "times": t}

        if "obd" in FLAGS.plots:
            logging.info("Plotting %s data..." % c)
            ax = axes[i]
            ax.plot(t, y, "-x")
            ax.set_title(c)
            ax.set_xlabel("time [s]")
            ax.set_ylabel("%s" % unit)
    if "obd" in FLAGS.plots:
        plt.tight_layout()
        if FLAGS.plot_mode == "save":
            plot_path = os.path.join(plots_dir, "odb.pdf")
            logging.info("Saving plot to '%s'...", plot_path)
            plt.savefig(plot_path)
            plt.close()
    
    # Estimate f0 from audio (if not already done)
    f0_frame_size = _CREPE_FRAME_SIZE // 2#16
    f0_frame_rate = _CREPE_SAMPLE_RATE / f0_frame_size # frame rate of f0 and loudness features
    f0_path = os.path.join(FLAGS.data_dir, "f0.npy")
    audio_path = os.path.join(FLAGS.data_dir, FLAGS.audio_filename)
    audio, _ = librosa.load(audio_path, FLAGS.sample_rate)
    if os.path.exists(f0_path):
        logging.info("Using precomputed f0.")
        f0 = np.load(f0_path)
    else:
        logging.info("Estimating f0 using CREPE...")
        audio_lores = librosa.core.resample(audio, FLAGS.sample_rate, _CREPE_SAMPLE_RATE)
        f0, _ = compute_f0(audio_lores, _CREPE_SAMPLE_RATE, f0_frame_rate)
        np.save(f0_path, f0)
    
    # Interpolate OBD quantities to allow for upsampling later
    c_interp = dict()
    for c in FLAGS.commands:
        logging.info("Interpolating %s signal..." % c)
        c_interp[c] = interp1d(c_dict[c]["times"], c_dict[c]["values"], kind=FLAGS.interp_method)

    # Scale interpolated RPM signal to become f0 and sample on CREPE f0 times
    logging.info("Upsampling RPM signal...")
    f0_times = np.arange(0., len(f0)/f0_frame_rate, 1/f0_frame_rate)
    f0_rpm_times = f0_times[f0_times < np.max(c_dict["RPM"]["times"])]
    rpm_scale = 1./60
    if FLAGS.engine_type == "four-stroke":
        rpm_scale *= 2
    f0_rpm = rpm_scale*c_interp["RPM"](f0_rpm_times)

    # Find lag between f0 and RPM
    logging.info("Calculating lag between f0 and RPM...")
    xcorr = correlate(f0 - np.mean(f0), f0_rpm - np.mean(f0_rpm))
    lag = xcorr.argmax() - (len(f0_rpm) - 1)
    logging.info("Found lag: %d frames (%.3f seconds)" % (lag, lag/f0_frame_rate))
    if "f0-rpm" in FLAGS.plots:
        logging.info("Plotting RPM alignment...")
        _, axes = plt.subplots(2, 1, figsize=(15, 6))
        axes[0].plot(f0_times, f0, label="f0")
        axes[0].plot(f0_rpm_times, f0_rpm, label="f0-rpm")
        axes[0].set_title("Before alignment")
        axes[1].plot(f0_times - lag/f0_frame_rate, f0, label="f0")
        axes[1].plot(f0_rpm_times, f0_rpm, label="f0-rpm")
        axes[1].set_title("After alignment")
        plt.tight_layout()
        if FLAGS.plot_mode == "save":
            plot_path = os.path.join(plots_dir, "f0_rpm.pdf")
            logging.info("Saving plot to '%s'...", plot_path)
            plt.savefig(plot_path)
            plt.close()
    
    # Trim audio according to lag
    start = int(lag*FLAGS.sample_rate/f0_frame_rate)
    end = start + int(len(f0_rpm)*FLAGS.sample_rate/f0_frame_rate) + 1
    audio_trimmed = audio[start:end]
    audio_trimmed_length = len(audio_trimmed)/float(FLAGS.sample_rate)
    logging.info("Trimmed audio is %.3f seconds." % audio_trimmed_length)
    if "f0-audio" in FLAGS.plots:
        logging.info("Plotting audio spectrogram with f0...")
        fmax = 2**13
        n_fft = 2**13
        plt.figure(figsize=(15, 6))
        S = librosa.feature.melspectrogram(y=audio_trimmed, sr=FLAGS.sample_rate, n_fft=n_fft, n_mels=1024, fmax=fmax)
        S_dB = librosa.power_to_db(S, ref=np.max)
        ax = specshow(S_dB, x_axis='time', y_axis='mel', sr=FLAGS.sample_rate, fmax=fmax)
        f0_h, = ax.plot(f0_rpm_times, f0_rpm, "--")
        ax.set_ylim((0, 5*np.max(f0_rpm)))
        ax.set_xlabel("time [s]")
        ax.set_ylabel("frequency [Hz]")
        ax.legend([f0_h], ["synched f0 from RPM"], loc="upper right")
        plt.tight_layout()
        if FLAGS.plot_mode == "save":
            plot_path = os.path.join(plots_dir, "f0_audio.pdf")
            logging.info("Saving plot to '%s'...", plot_path)
            plt.savefig(plot_path)
            plt.close()
    
    # Resample OBD signals and store together with audio in dict
    logging.info("Resampling input signals to given frame rate...")
    input_times = np.arange(0., np.max(c_dict["RPM"]["times"]), 1/FLAGS.frame_rate)
    f0_signal = rpm_scale*c_interp["RPM"](input_times)
    data = {
        "sample_rate": FLAGS.sample_rate,
        "frame_rate": FLAGS.frame_rate,
        "audio": audio_trimmed,
        "inputs": {"f0": f0_signal}
    }
    for c in FLAGS.commands:
        data["inputs"][c] = c_interp[c](input_times)
    audio_times = np.arange(0., audio_trimmed_length, 1./FLAGS.sample_rate)
    data_path = os.path.join(save_dir, "data.pickle")
    logging.info("Saving data to %s..." % data_path)
    pickle.dump(data, open(data_path, "wb"))
    if "data" in FLAGS.plots:
        if FLAGS.plot_mode == "save":
            plot_path = os.path.join(plots_dir, "data.pdf")
            logging.info("Plotting and saving to '%s'...", plot_path)
            plot_data_dict(data, save_path=plot_path)
            plt.close()
    
    if FLAGS.plot_mode == "show":
        plt.show()
    logging.info("Done.")

if __name__ == "__main__":
    app.run(main)