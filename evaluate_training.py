import os
from absl import app, logging, flags

from evaluation.training import TrainingEvaluator

FLAGS = flags.FLAGS
flags.DEFINE_string("data_name", None,
                    "Name of the training data (name of training data directory under ./evaluation/fig/training).")
flags.DEFINE_float("smooth", 0.99,
                   "Smoothing factor.")
flags.DEFINE_float("iter_per_epoch", 675.0/32.0,
                     "Number of iterations per epoch of given dataset.")
flags.DEFINE_list("ylim", None,
                  "y-axis limits of plots.")
flags.DEFINE_bool("show", False,
                  "Whether to show plots.")
flags.DEFINE_string("split", "all",
                    "Splits to plot (all, train, valid).")

# Define global constants
FIG_BASE_DIR = os.path.join(".", "evaluation", "fig", "training")
FIG_EXTENSION = "pdf"

def main(argv):
    # Check preconditions
    if FLAGS.ylim:
        FLAGS.ylim = [float(FLAGS.ylim[i]) for i in range(len(FLAGS.ylim))]

    # Initialize training evaluator
    evaluator = TrainingEvaluator(FLAGS.data_name)

    # Plot individual loss curves
    fig_dir = os.path.join(FIG_BASE_DIR, FLAGS.data_name)
    for model_id in evaluator.data:
        fig_path = os.path.join(fig_dir, "%s.%s" % (model_id, FIG_EXTENSION))
        evaluator.plot_losses(model_ids=[model_id],
                              smooth=FLAGS.smooth,
                              iter_per_epoch=FLAGS.iter_per_epoch,
                              ylim=FLAGS.ylim,
                              plot_split=FLAGS.split,
                              show=FLAGS.show,
                              save_path=fig_path)

    # Plot all loss curves
    fig_path = os.path.join(fig_dir, "all." + FIG_EXTENSION)
    evaluator.plot_losses(smooth=FLAGS.smooth, 
                          plot_split=FLAGS.split,
                          show=FLAGS.show,
                          save_path=fig_path)

if __name__ == "__main__":
    app.run(main)