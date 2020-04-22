'''
Script for evaluating data.
'''
import os
from absl import app, flags, logging

from data.data_provider import TFRecordProvider
from evaluation.data import DataEvaluator

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", os.path.join(".", "data", "tfrecord", "ford_final"),
                    "Path to directory of TFRecord dataset.")
flags.DEFINE_bool("show", False,
                  "Whether to show the plots.")
flags.DEFINE_bool("f0_hist", False,
                  "Whether to plot f0 histograms.")
flags.DEFINE_bool("ex_figures", False,
                  "Whether to produce figures illustrating individual examples " + \
                  "(Spectrograms and input plots).")
flags.DEFINE_list("ex_ids_train", [1, 127, 78],
                  "Training examples to illustrate (if ex_figures is True).")
flags.DEFINE_list("ex_ids_test", [1, 42, 49],
                  "Test examples to illustrate (if ex_figures is True).")
flags.DEFINE_list("ex_input_keys", ["f0", "phase_sub_sync"],
                  "Input keys to show in example illustrations (if ex_figures is True).")

# Define global constants
FIG_BASE_DIR = os.path.join(".", "evaluation", "fig")
FIG_EXTENSION = "pdf"

def main(argv):
    # Check preconditions
    if FLAGS.data_dir is None:
        raise ValueError("data_dir must be set.")
    
    # Initiate values that are constant for each split
    constants = {"f0_hist_xlim": None,
                 "f0_hist_ylim": None,
                 "f0_hist_bins": None,
                 "f0_hist_yscale": "linear",
                 "f0_diff_hist_xlim": None,
                 "f0_diff_hist_ylim": None,
                 "f0_diff_hist_bins": None,
                 "f0_diff_hist_yscale": "log",
                 "ex_ids": {"train": FLAGS.ex_ids_train,
                            "test": FLAGS.ex_ids_test}}

    for split in DataEvaluator.SPLITS:
        # Load data and initialize data evaluator
        data_provider = TFRecordProvider(FLAGS.data_dir, split=split)
        data_evaluator = DataEvaluator(data_provider)
        fig_dir = os.path.join(FIG_BASE_DIR, data_evaluator.dataset_name)
        fig_data_dir = os.path.join(fig_dir, "data")

        # Plot f0 histograms
        if FLAGS.f0_hist:
            # Get data
            f0_hist_data_path = os.path.join(fig_data_dir, "f0_hist_%s_data.pickle" % split)
            if not os.path.exists(f0_hist_data_path):
                # Calculate f0 histogram data and save for later
                logging.info("Calculating f0 histogram data for '%s'..." % data_evaluator.dataset_name)
                f0_hist_data = data_evaluator.get_f0_hist_dict(
                    bins_f0=constants["f0_hist_bins"],
                    bins_f0_diff=constants["f0_diff_hist_bins"])
                logging.info("Saving to '%s'..." % f0_hist_data_path)
                data_evaluator.save_data_to_file(f0_hist_data, f0_hist_data_path)
                if constants["f0_hist_bins"] is None:
                    constants["f0_hist_bins"] = f0_hist_data["f0_hist_bins"]
                    constants["f0_diff_hist_bins"] = f0_hist_data["f0_diff_hist_bins"]
            else:
                # Get pre-calculated data from file
                logging.info("Loading pre-calculated data from '%s'..." % f0_hist_data_path)
                f0_hist_data = data_evaluator.load_data_from_file(f0_hist_data_path)
            
            # Plot f0 histograms
            input_keys = ["f0", "f0_diff"]
            for i, input_key in enumerate(input_keys):
                fig_path = os.path.join(fig_dir, "%s_hist_%s.%s" % (input_key, split, FIG_EXTENSION))
                logging.info("Plotting %s histogram..." % input_key)
                xlim, ylim = data_evaluator.plot_hist_from_dict(
                    f0_hist_data,
                    input_key=input_key,
                    norm=True,
                    show=FLAGS.show,
                    save_path=fig_path,
                    split=split,
                    xlabel="fundamental frequency $f_0$ [Hz]",
                    xlim=constants[input_key + "_hist_xlim"],
                    ylim=constants[input_key + "_hist_ylim"],
                    yscale=constants[input_key + "_hist_yscale"])
                if constants[input_key + "_hist_xlim"] is None:
                    constants[input_key + "_hist_xlim"] = xlim
                    constants[input_key + "_hist_ylim"] = ylim
        
        # Plot Mel spectrograms and input plots
        if FLAGS.ex_figures and \
           split not in ["all", "valid"]: # only use train and test examples
            # For each example: get data, plot inputs, plot Mel spectrogram
            for ex_id in constants["ex_ids"][split]:
                # Get data
                logging.info("Getting input data data for %s example %d of '%s'..." % \
                             (split, ex_id, data_evaluator.dataset_name))
                ex_dict = data_evaluator.get_example_dict(ex_id, FLAGS.ex_input_keys)

                # Input plots
                fig_path = os.path.join(fig_dir, "ex_inputs_%s_%d.%s" % (split, ex_id, FIG_EXTENSION))
                logging.info("Plotting inputs for %s example %d of '%s'..." % \
                             (split, ex_id, data_evaluator.dataset_name))
                data_evaluator.plot_inputs_from_dict(ex_dict, split, save_path=fig_path, show=FLAGS.show)

                # Spectrograms
                logging.info("Plotting spectrograms for %s example %d of '%s'..." % \
                             (split, ex_id, data_evaluator.dataset_name))
                for spec_type in ["mel", "cqt"]:
                    fig_path = os.path.join(fig_dir, "ex_%s_%d_%s.%s" % \
                                            (split, ex_id, spec_type, FIG_EXTENSION))
                    data_evaluator.plot_spectrogram_from_dict(ex_dict, spec_type=spec_type,
                                                                       plot_f0=True,
                                                                       split=split,
                                                                       save_path=fig_path,
                                                                       show=FLAGS.show)

    logging.info("Done!")

if __name__ == "__main__":
    app.run(main)
