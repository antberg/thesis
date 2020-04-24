from absl import app, logging, flags

from evaluation.training import TrainingEvaluator

FLAGS = flags.FLAGS
flags.DEFINE_string("data_name", None,
                    "Name of the training data (name of training data directory under ./evaluation/fig/training).")

def main(argv):
    # Initialize training evaluator
    evaluator = TrainingEvaluator(FLAGS.data_name)

    # Plot individual loss curves
    for model_id in evaluator.data:
        evaluator.plot_losses(model_ids=[model_id])

    # Plot all loss curves
    evaluator.plot_losses()

if __name__ == "__main__":
    app.run(main)