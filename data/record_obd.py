'''
Script for recording OBD-II data.
'''
import obd
import time
import os
import pickle
from absl import app, flags

from util import TimedList, get_timestamp

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "./raw/obd", "Data directory where to store recordings.")
flags.DEFINE_list("commands", ["RPM", "SPEED", "THROTTLE_POS", "RUN_TIME"], "Comma-separated list of OBD commands to subscribe to.")
flags.DEFINE_bool("debug", True, "Whether to print debug log.")
flags.DEFINE_bool("audio", False, "Whether to record audio while recording OBD data.")
flags.DEFINE_integer("length", 1, "Length of recording in seconds.")
flags.DEFINE_integer("rate", 10, "Update rate of OBD commands in milliseconds.")

def get_callback_handle(l):
    '''Get callback handle for OBD subscribers.'''
    return lambda r: l.append(r.value)

def main(argv):
    if FLAGS.debug:
        print("Will print debug log.")
        obd.logger.setLevel(obd.logging.DEBUG)

    # Connect to ELM327 interface
    print("Connecting to ELM327 interface...")
    conn = obd.Async(delay_cmds=FLAGS.rate/1e3)

    # Create folder
    print("Creating data folder...")
    timestamp = get_timestamp()
    write_dir = os.path.join(FLAGS.data_dir, timestamp)
    os.makedirs(write_dir)

    # Subscribe to given commands and store results in lists
    print("Recording OBD data...")
    c_dict = dict()
    c_callbacks = dict()
    for c in FLAGS.commands:
        c_dict[c] = TimedList(name=c, debug=FLAGS.debug)
        c_callbacks[c] = get_callback_handle(c_dict[c])
        conn.watch(c, callback=c_callbacks[c])
    conn.start()
    time.sleep(FLAGS.length)
    conn.stop()

    # Record audio
    if FLAGS.audio:
        raise NotImplementedError("Audio recording has not been implemented yet.")

    # Store recorded values
    print("Storing recorded values...")
    for c in FLAGS.commands:
        with open(os.path.join(write_dir, "%s.pickle" % c), "wb") as f:
            pickle.dump(c_dict[c], f)
    
    print("Done.")

if __name__ == "__main__":
    app.run(main)