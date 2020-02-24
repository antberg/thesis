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

from util import plot_audio_f0, get_timestamp
from constants import DEFAULT_WINDOW_SECS, DEFAULT_HOP_SECS

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", None, "Directory of training data.")
flags.DEFINE_string("tfrecord_path", None, "Save path of created TFRecord.")
flags.DEFINE_integer("window_secs", DEFAULT_WINDOW_SECS, "Length of training windows in seconds.")
flags.DEFINE_integer("hop_secs", DEFAULT_HOP_SECS, "Size of training hops in seconds.")
flags.DEFINE_bool("shuffle", False, "Shuffle the windows.")
flags.DEFINE_bool("inspect_windows", False, "Inspect each window manually for debugging.")

def get_float_feature(value):
    '''Returns a float_list from a float / double.'''
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def get_int64_feature(value):
    '''Returns an int64_list from a bool / enum / int / uint.'''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_n_windows(sequence, rate, window_secs, hop_secs):
    window_size = int(window_secs * rate)
    hop_size = int(hop_secs * rate)
    return int(np.ceil((len(sequence) - window_size) / hop_size)) + 1

def split_data(data, window_secs, hop_secs, shuffle=True):
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
        n_windows = int(np.ceil((len(sequence) - window_size) / hop_size)) + 1
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
            n_windows = get_n_windows(data["audio"], audio_rate, window_secs, hop_secs)
            n_samples += n_windows
            for i, window in enumerate(split_data(data, window_secs, hop_secs, FLAGS.shuffle)):
                logging.info("Processing window %d out of %d..." % (i+1, n_windows))
                writer.write(get_serialized_example(window))
                if FLAGS.inspect_windows:
                    plt.figure(figsize=(8, 4))
                    plot_audio_f0(window["audio"], audio_rate, window["f0"], input_rate)
                    sd.play(window["audio"], audio_rate)
                    plt.show()
                    sd.wait()
            logging.info("Done with '%s'." % data_path)
    
    metadata = {
        "audio_rate": audio_rate,
        "input_rate": input_rate,
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