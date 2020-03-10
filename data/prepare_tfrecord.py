'''
Prepare TFRecords from preprocessed data files.

This script produces a TFRecord dataset containg audio and f0 features.
'''
import os
import sys
import glob
import pickle
import random
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
import matplotlib.pyplot as plt
import sounddevice as sd
from ddsp.core import oscillator_bank
from scipy.interpolate import interp1d
from scipy.signal import correlate

from util import plot_audio_f0, get_timestamp, pass_filter
from constants import DEFAULT_WINDOW_SECS, DEFAULT_HOP_SECS

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", None, "Directory of training data.")
flags.DEFINE_string("tfrecord_path", None, "Save path of created TFRecord.")
flags.DEFINE_integer("window_secs", DEFAULT_WINDOW_SECS, "Length of training windows in seconds.")
flags.DEFINE_integer("hop_secs", DEFAULT_HOP_SECS, "Size of training hops in seconds.")
flags.DEFINE_bool("shuffle", False, "Shuffle the windows.")
flags.DEFINE_bool("inspect_windows", False, "Inspect each window manually for debugging.")
flags.DEFINE_bool("osc", False, "Whether to generate a synchronized oscillating signal.")
flags.DEFINE_bool("pad", False, "Whether to pad the end of the recording with zeros.")

def get_float_feature(value):
    '''Returns a float_list from a float / double.'''
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def get_int64_feature(value):
    '''Returns an int64_list from a bool / enum / int / uint.'''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _oscillator_bank(f0, sample_rate):
    '''Return an oscillator signal corresponding to f0.'''
    N = np.size(f0)
    f0 = f0.reshape((1, N, 1))
    a = np.ones((1, N, 1))
    y = oscillator_bank(f0, a, sample_rate)
    return y.numpy()[0,:]

def _resample(y, fs_old, fs_new, interp_method="linear"):
    '''Resample a signal through interpolation'''
    y = np.append(y, y[-1])
    N = np.size(y)
    T = N/fs_old
    t_old = np.arange(0.0, T, 1/fs_old)[:N]
    y_interp = interp1d(t_old, y, kind=interp_method)
    t_new = np.arange(0.0, T-1/fs_old, 1/fs_new)
    return y_interp(t_new)

def get_synchronized_osc(audio, f0, sample_rate, frame_rate,
                                                 window_secs=.1,
                                                 hop_secs=.05,
                                                 f_cutoff=200):
    '''
    Generate an oscillating signal based on f0 phase-synchronized with the
    audio.
    '''
    # Low-pass filter audio
    audio_lopass = pass_filter(audio, sample_rate, f_cutoff, btype="low")
    
    # Upsample f0 to audio sample rate to get maximum-precision lag
    n_samples = np.size(audio)
    audio_secs = n_samples/sample_rate
    f0_upsample = _resample(f0, frame_rate, sample_rate)

    # For each window, generate oscillating signal, synchronize and crop
    window_size = int(window_secs * sample_rate)
    hop_size = int(hop_secs * sample_rate)
    n_windows = int(np.ceil((n_samples - window_size) / hop_size)) + 1
    n_samples_padded = (n_windows - 1) * hop_size + window_size
    n_padding = n_samples_padded - n_samples
    audio_lopass = np.pad(audio_lopass, (0, n_padding), mode="constant")
    f0_upsample = np.pad(f0_upsample,
                         (0, n_padding),
                         mode="constant",
                         constant_values=(0, f0_upsample[-1]))
    windows_ends = range(window_size, n_samples + 1, hop_size)
    osc = np.zeros((n_samples,))
    for i, window_end in enumerate(windows_ends):
        # Get audio window
        audio_window = audio_lopass[window_end-window_size:window_end]

        # Generate oscillating signal, one period longer than window
        n_period = int(sample_rate/f0_upsample[window_end-1])
        if window_end + n_period > n_samples:
            f0_window = np.pad(f0_upsample[window_end-window_size:window_end],
                               (0, n_period),
                               mode="constant",
                               constant_values=(0, f0_upsample[window_end-1]))
        else:
            f0_window = f0_upsample[window_end-window_size:window_end+n_period]
        osc_window = _oscillator_bank(f0_window, sample_rate)

        # Synchronize osc and audio
        xcorr = correlate(osc_window - np.mean(osc_window),
                          audio_window - np.mean(audio_window))
        lag = xcorr[window_size-1:window_size+n_period-1].argmax()
        osc_window = osc_window[lag:lag+window_size]
        
        # Cross-fade osc windows
        if i == 0:
            osc[:window_size] = osc_window
        else:
            osc[window_end-window_size:window_end-hop_size] = \
                .5 * osc[window_end-window_size:window_end-hop_size] + \
                .5 * osc_window[:window_size-hop_size]
            osc[window_end-hop_size:window_end] = osc_window[window_size-hop_size:]
    osc = _resample(osc, sample_rate, frame_rate)
    return osc

def get_n_windows(sequence, rate, window_secs, hop_secs, pad=False):
    window_size = int(window_secs * rate)
    hop_size = int(hop_secs * rate)
    n_windows = int(np.ceil((len(sequence) - window_size) / hop_size))
    if pad:
        n_windows += 1
    return n_windows

def split_data(data, window_secs, hop_secs, shuffle=False, generate_osc=False, pad=False):
    '''
    Generator function for generating windows of training examples.
    Inspired by https://github.com/magenta/ddsp/blob/master/ddsp/training/data_preparation/prepare_tfrecord_lib.py
    '''
    if shuffle:
        seed = random.randrange(sys.maxsize)

    def get_windows(sequence, rate):
        '''Generate a single window of a sequence'''
        window_size = int(window_secs * rate)
        hop_size = int(hop_secs * rate)
        n_windows = int(np.ceil((len(sequence) - window_size) / hop_size))
        if pad:
            n_windows += 1
            n_samples_padded = (n_windows - 1) * hop_size + window_size
            n_padding = n_samples_padded - len(sequence)
            sequence = np.pad(sequence, (0, n_padding), mode="constant")
        windows_ends = range(window_size, len(sequence) + 1, hop_size)
        if shuffle:
            windows_ends = list(windows_ends)
            random.Random(seed).shuffle(windows_ends)
        for window_end in windows_ends:
            yield sequence[window_end-window_size:window_end]
    
    for audio, f0 in zip(get_windows(data["audio"], data["sample_rate"]),
                         get_windows(data["inputs"]["f0"], data["frame_rate"])):
        if generate_osc:
            osc = get_synchronized_osc(audio, f0, data["sample_rate"], data["frame_rate"])
            yield {"audio": audio, "f0": f0, "osc": osc}
        else:
            yield {"audio": audio, "f0": f0}

def get_serialized_example(data):
    '''Get serialized tf.train.Example from dictionary of floats.'''
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                k: tf.train.Feature(float_list=tf.train.FloatList(value=v))
                for k, v in data.items()
            }
    ))
    return example.SerializeToString()

def main(argv):
    # Check preconditions
    if FLAGS.data_dir is None:
        raise ValueError("Data directory must be set (using the --data_dir flag).")
    tfrecord_path = FLAGS.tfrecord_path
    if tfrecord_path is None:
        tfrecord_path = os.path.join("./tfrecord", get_timestamp(), "data.tfrecord")
        logging.info("No save path for the TFRecord specified, will save to '%s'." % tfrecord_path)
    elif os.path.exists(tfrecord_path):
        raise FileExistsError("'%s' already exists." % tfrecord_path)
    tfrecord_dir = os.path.dirname(tfrecord_path)
    if not os.path.exists(tfrecord_dir):
        os.makedirs(tfrecord_dir)

    logging.info("Will load pickled data from '%s'." % FLAGS.data_dir)
    data_path_format = os.path.join(FLAGS.data_dir, "*.pickle")
    window_secs = FLAGS.window_secs
    hop_secs = FLAGS.hop_secs
    logging.info("Will split training examples into %d second windows with %d second hops." % (window_secs, hop_secs))

    audio_rate = -1
    input_rate = -1
    n_samples = 0
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for data_path in glob.glob(data_path_format):
            # Load example
            logging.info("Loading '%s'..." % data_path)
            with open(data_path, "rb") as data_file:
                data = pickle.load(data_file)
            
            # Make sure sample rates are consistent
            if audio_rate == -1:
                audio_rate = data["sample_rate"]
                input_rate = data["frame_rate"]
            elif audio_rate != data["sample_rate"]:
                raise ValueError("Data must have the same audio sample rate.")
            elif input_rate != data["frame_rate"]:
                raise ValueError("Data must have the same input sample rate.")
            
            # Split example into windows and write to TFRecord
            logging.info("Splitting into windows and writing to TFRecord...")
            n_windows = get_n_windows(data["audio"], audio_rate, window_secs, hop_secs, FLAGS.pad)
            n_samples += n_windows
            for i, window in enumerate(split_data(data, window_secs, hop_secs, FLAGS.shuffle, FLAGS.osc, FLAGS.pad)):
                logging.info("Processing window %d out of %d..." % (i+1, n_windows))
                writer.write(get_serialized_example(window))
                if FLAGS.inspect_windows:
                    plt.figure(figsize=(8, 4))
                    plot_audio_f0(window["audio"], audio_rate, window["f0"], input_rate)
                    sd.play(window["audio"], audio_rate)
                    plt.show()
                    sd.wait()
            logging.info("Done with '%s'." % data_path)
    
    input_keys = ["f0"]
    if FLAGS.osc:
        input_keys.append("osc")
    metadata = {
        "audio_rate": audio_rate,
        "input_rate": input_rate,
        "input_keys": input_keys,
        "n_samples": n_samples,
        "example_secs": window_secs,
        "hop_secs": hop_secs
    }
    metadata_path = os.path.join(tfrecord_dir, "metadata.pickle")
    print("Saving metadata to '%s'..." % metadata_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print("Done.")

if __name__ == "__main__":
    app.run(main)