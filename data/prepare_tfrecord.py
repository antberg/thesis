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
flags.DEFINE_string("tfrecord_name", None, "Name of TFRecords dataset.")
flags.DEFINE_integer("window_secs", DEFAULT_WINDOW_SECS, "Length of training windows in seconds.")
flags.DEFINE_integer("hop_secs", DEFAULT_HOP_SECS, "Size of training hops in seconds.")
flags.DEFINE_bool("shuffle", False, "Shuffle the windows.")
flags.DEFINE_bool("inspect_windows", False, "Inspect each window manually for debugging.")
flags.DEFINE_bool("osc", False, "Whether to generate a synchronized oscillating signal.")
flags.DEFINE_bool("pad", False, "Whether to pad the end of the recording with zeros.")
flags.DEFINE_list("split", [80, 10, 10], "Train-validation-test split of data.")

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

def _get_first_extremum(y):
    mono = np.diff(y) < 0
    extr_type = "min"
    if mono[0]:
        mono = ~mono
        extr_type = "max"
    return np.argmax(mono), extr_type

def _get_second_extremum(y):
    i, _ = _get_first_extremum(y)
    i_new, extr_type = _get_first_extremum(y[i:])
    return i+i_new, extr_type

def _sine_stitch(y1, y2):
    '''Stitch together two sines'''
    y = np.concatenate((y1, y2))

    # Find last peak/trough of y1
    i1, type1 = _get_first_extremum(np.flip(y1))
    i1 = len(y1) - i1 - 1

    # Find second peak/trough of y2
    i2, type2 = _get_second_extremum(y2)
    i2 = len(y1) + i2

    # Create sine stitch
    n_samples = i2 - i1
    n = np.arange(0, n_samples)
    period = n_samples if type1 == type2 else 2*n_samples
    phi = -np.pi/2 if type1 == "max" else np.pi/2
    stitch = np.sin(2*np.pi*n/period + phi)
    y[i1:i2] = stitch
    return y

def get_synchronized_osc(audio, f0, sample_rate, frame_rate,
                                                 window_secs=.05,
                                                 hop_secs=.05,
                                                 f_cutoff=None):
    '''
    Generate an oscillating signal based on f0 phase-synchronized with the
    audio.
    '''
    # Low-pass filter audio
    if f_cutoff is None:
        f_cutoff = 1.5*np.max(f0)
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
        
        # Stitch together osc windows
        if i == 0:
            osc[:window_size] = osc_window
        else:
            osc[window_end-2*window_size:window_end] = _sine_stitch(
                osc[window_end-2*window_size:window_end-window_size], 
                osc_window)
            #osc[window_end-window_size:window_end] = osc_window
    
    '''t_audio = np.arange(0.0, len(audio_lopass))/sample_rate
    plt.plot(t_audio, audio_lopass)
    osc = osc > 0
    t_osc = np.arange(0.0, len(osc))/frame_rate
    plt.plot(t_osc, np.max(np.abs(audio_lopass))*osc)
    plt.show()'''
    osc = _resample(osc, sample_rate, frame_rate)

    return osc

def get_n_windows(sequence, rate, window_secs, hop_secs, pad=False):
    window_size = int(window_secs * rate)
    hop_size = int(hop_secs * rate)
    n_windows = int(np.ceil((len(sequence) - window_size) / hop_size))
    if pad:
        n_windows += 1
    return n_windows

def generate_windows(data, window_secs, hop_secs, shuffle=False, generate_osc=False, pad=False):
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

def split_ids(n_windows_total, splits):
    n_windows = {s: 0 for s in ["total", "train", "valid", "test"]}
    n_windows["total"] = n_windows_total
    for split in ["valid", "test"]:
        n_windows[split] = int(splits[split] * n_windows["total"])
    n_windows["train"] = n_windows["total"] - n_windows["valid"] - n_windows["test"]
    ids = dict()
    ids["total"] = list(range(n_windows["total"]))
    random.shuffle(ids["total"])
    ids_begin = 0
    for split in ["train", "valid", "test"]:
        ids_end = ids_begin + n_windows[split]
        ids[split] = ids["total"][ids_begin:ids_end]
        ids_begin = ids_end
    return ids

def get_split_from_id(i, ids):
    for split in ["train", "valid", "test"]:
        if i in ids[split]:
            return split
    raise ValueError("%d is in none of the splits!" % i)

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
    if len(FLAGS.split) != 3:
        raise ValueError("split must have 3 elements (train, valid, test).")
    splits = {split: float(percent) / 100
              for split, percent in zip(["train", "valid", "test"], FLAGS.split)}
    if sum(splits.values()) != 1.0:
        raise ValueError("split must sum to 100.")
    if FLAGS.hop_secs < FLAGS.window_secs and splits["train"] < 1.0:
        raise ValueError("Windows must be disjoint (i.e. hop_secs >= window_secs) " +\
                         "when splitting data into train-valid-test or there " +\
                         "will be training samples in valid-test datasets.")

    # Set paths to .tfrecord files
    tfrecord_name = get_timestamp() if FLAGS.tfrecord_name is None else FLAGS.tfrecord_name
    tfrecord_dir = os.path.join("./tfrecord", tfrecord_name)
    if not os.path.exists(tfrecord_dir):
        os.makedirs(tfrecord_dir)
    tfrecord_paths = {split: os.path.join(tfrecord_dir, "%s.tfrecord" % split)
                      for split in ["train", "valid", "test"]}
    logging.info("Will save TFRecords to '%s'." % tfrecord_dir)

    # Paths to data
    logging.info("Will load pickled data from '%s'." % FLAGS.data_dir)
    data_path_format = os.path.join(FLAGS.data_dir, "*.pickle")

    # Split each data file into windows and write to .tfrecord files
    audio_rate = -1
    input_rate = -1
    n_samples = {split: 0 for split in ["total", "train", "valid", "test"]}
    window_secs = FLAGS.window_secs
    hop_secs = FLAGS.hop_secs
    logging.info("Will split training examples into %d second windows with %d second hops." % (window_secs, hop_secs))
    with tf.io.TFRecordWriter(tfrecord_paths["train"]) as writer_train, \
         tf.io.TFRecordWriter(tfrecord_paths["valid"]) as writer_valid, \
         tf.io.TFRecordWriter(tfrecord_paths["test"]) as writer_test:
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

            # Split samples into train, valid and test
            n_windows_total = get_n_windows(data["audio"], audio_rate, window_secs, hop_secs, FLAGS.pad)
            ids = split_ids(n_windows_total, splits)
            n_windows = {s: len(i) for s, i in ids.items()}
            for split in ["total", "train", "valid", "test"]:
                n_samples[split] += n_windows[split]
            logging.info("Will use %s%%, %s%%, %s%% splits for train, valid, test." % tuple(FLAGS.split))
            logging.info("Will split %d windows into %d, %d, %d windows for train, valid, test." % tuple(n_windows.values()))
            
            # Split example into windows and write to .tfrecord files
            logging.info("Splitting into windows and writing to .tfrecord files...")
            for i, window in enumerate(generate_windows(data, window_secs, hop_secs, FLAGS.shuffle, FLAGS.osc, FLAGS.pad)):
                # Write window to given split
                split = get_split_from_id(i, ids)
                logging.info("Processing window %d out of %d... (%s)" % (i+1, n_windows["total"], split))
                serialized_example = get_serialized_example(window)
                if split == "train":
                    writer_train.write(serialized_example)
                elif split == "valid":
                    writer_valid.write(serialized_example)
                elif split == "test":
                    writer_test.write(serialized_example)

                # Inspect window, if applicable
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
        "n_samples": n_samples["total"],
        "n_samples_train": n_samples["train"],
        "n_samples_valid": n_samples["valid"],
        "n_samples_test": n_samples["test"],
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