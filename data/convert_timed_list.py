'''
Script for converting previously used lists to TimedLists.
'''
import os
import pickle
from absl import app, flags

from util import TimedList

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "./raw/obd/ford/1min_old", "Data directory.")
flags.DEFINE_list("commands", ["RPM", "SPEED", "THROTTLE_POS"], "Commands corresponding to recorded quantities.")
flags.DEFINE_string("time_unit", "ns", "Unit of time stamps (e.g. 's', 'ms', 'ns', etc.).")

def preprocess_times(times):
    '''Make sure time units is seconds.'''
    denom = 0.
    if FLAGS.time_unit == "ns":
        denom = 1e9
    else:
        raise NotImplementedError("Time unit %s not supported" % FLAGS.time_unit)

    for i, t in enumerate(times):
        times[i] = t/denom

    return times

def main(argv):
    print("Converting to timed lists...")
    pickle_path = os.path.join(FLAGS.data_dir, "time.pickle")
    times = pickle.load(open(pickle_path, "rb"))
    times = preprocess_times(times)
    data_dir_new = os.path.join(FLAGS.data_dir, "new")
    if not os.path.exists(data_dir_new):
        os.makedirs(data_dir_new)
    for c in FLAGS.commands:
        pickle_path = os.path.join(FLAGS.data_dir, c + ".pickle")
        print("Unpickling '%s'..." % pickle_path)
        values = pickle.load(open(pickle_path, "rb"))
        pickle_path_new = os.path.join(data_dir_new, c + ".pickle")
        print("Pickling '%s'..." % pickle_path_new)
        timed_list = TimedList(name=c, values=values, times=times)
        pickle.dump(timed_list, open(pickle_path_new, "wb"))
    print("Done.")

if __name__ == "__main__":
    app.run(main)