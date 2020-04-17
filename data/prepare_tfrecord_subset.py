'''
Prepare subsets of existing TFRecord datasets.

This script produces a TFRecord dataset whose training examples are a subset of the
original TFRecord dataset.

This is a script inside a module and must therefore be run from the parent directory
(..) as follows:
    python -m data.prepare_tfrecord_subset [--args values]
'''
import os
import shutil
import pickle
from absl import app, flags, logging
import matplotlib.pyplot as plt
import tensorflow as tf

from .data_provider import TFRecordProvider
from .util import get_serialized_example_tf

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", None, "Directory of TFRecord dataset.")
flags.DEFINE_string("subset_dir", None, "Directory to new subset TFRecord dataset.")
flags.DEFINE_list("ex_ids", [], "IDs of examples from training split to use in subset.")
flags.DEFINE_bool("inspect", False, "Whether to plot an playback selected examples.")

def main(argv):
    if FLAGS.data_dir is None:
        raise ValueError("data_dir must be set (e.g. --data_dir path/to/my/data).")
    elif FLAGS.subset_dir is None:
        FLAGS.subset_dir = FLAGS.data_dir + "_mini"
    if len(FLAGS.ex_ids) == 0:
        raise ValueError("ex_ids must be set (e.g. --ex_ids 2,4,56).")

    # Load data
    data_provider = TFRecordProvider(FLAGS.data_dir, split="train")
    data_iter = iter(data_provider.get_batch(1, shuffle=False, repeats=1))

    # Store examples to a new .tfrecord file
    if not os.path.exists(FLAGS.subset_dir):
        os.makedirs(FLAGS.subset_dir)
    tfrecord_path = os.path.join(FLAGS.subset_dir, "subset_train.tfrecord")
    if os.path.exists(tfrecord_path):
        raise ValueError("'%s' already exists!" % tfrecord_path)
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for i, ex in enumerate(data_iter):
            # Skip this example if its not among those specified
            if str(i+1) not in FLAGS.ex_ids:
                continue

            # Write example to .tfrecord file
            logging.info("Writing example number %d to file..." % (i+1))
            serialized_example = get_serialized_example_tf(ex)
            writer.write(serialized_example)

            # Inspect example, if specified
            if FLAGS.inspect:
                n_axes = 1 + len(data_provider.input_keys)
                fig, axes = plt.subplots(n_axes, 1, figsize=(10, n_axes*3))
                axes_iter = iter(axes)
                ax = next(axes_iter)
                ax.plot(ex["audio"].numpy()[0,:])
                ax.set_title("audio")
                for key in data_provider.input_keys:
                    ax = next(axes_iter)
                    ax.plot(ex[key].numpy()[0,:])
                    ax.set_title(key)
                fig.suptitle("Example %d" % (i+1))
                plt.show()
    
    # Copy validation and test .tfrecord files
    logging.info("Copying validation and test files...")
    for split in ["valid", "test"]:
        source = os.path.join(FLAGS.data_dir, split+".tfrecord")
        target = os.path.join(FLAGS.subset_dir, split+".tfrecord")
        shutil.copyfile(source, target)
    
    # Modify and save metadata
    logging.info("Saving metadata...")
    metadata = data_provider.metadata
    n_train_diff = metadata["n_samples_train"] - len(FLAGS.ex_ids)
    metadata["n_samples"] -= n_train_diff
    metadata["n_samples_train"] = len(FLAGS.ex_ids)
    metadata_path = os.path.join(FLAGS.subset_dir, "metadata.pickle")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    logging.info("Done! All was successfully saved to '%s'." % FLAGS.subset_dir)

if __name__ == "__main__":
    app.run(main)