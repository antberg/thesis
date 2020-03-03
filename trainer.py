'''
Script for training.
'''
import os
from absl import app, logging, flags
from ddsp.training.train_util import Trainer, train, get_strategy

from data_provider import TFRecordProvider
from model_builder import ModelBuilder
from models.losses import MelSpectralLoss

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_dir", None, "Directory to store checkpoints.")
flags.DEFINE_string("data_dir", None, "Directory of training data (TFRecords).")

def main(argv):
    if FLAGS.checkpoint_dir is None:
        ValueError("checkpoint_dir must be set.")
    if FLAGS.data_dir is None:
        ValueError("data_dir must be set.")

    summary_dir = os.path.join(FLAGS.checkpoint_dir, "summaries", "train")
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    logging.info("Loading data...")
    data_provider = TFRecordProvider(FLAGS.data_dir)

    logging.info("Building model...")
    n_cylinders = 4.
    model = ModelBuilder(
        model_type="f0_rnn_fc_hpn_decoder",
        audio_rate=data_provider.audio_rate,
        input_rate=data_provider.input_rate,
        window_secs=data_provider.example_secs,
        f0_denom=n_cylinders,
        checkpoint_dir=FLAGS.checkpoint_dir,
        losses=[MelSpectralLoss(
            sample_rate=data_provider.audio_rate,
            n_bands=2
        )]
    ).build()

    logging.info("Building trainer...")
    strategy = get_strategy()
    trainer = Trainer(model, strategy)

    logging.info("Initializing training...")
    batch_size = 32
    while True:
        try:
            train(data_provider, trainer, batch_size=batch_size, steps_per_summary=1, steps_per_save=10, model_dir=model_dir)
        except:
            logging.info("An error ocurred, reinitializing training from last checkpoint...")

if __name__ == "__main__":
    app.run(main)
