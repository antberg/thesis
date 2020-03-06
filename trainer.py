'''
Script for training.
'''
import os
import tensorflow as tf
from absl import app, logging, flags
from ddsp.training.train_util import Trainer, write_gin_config, get_strategy

from data_provider import TFRecordProvider
from model_builder import ModelBuilder
from models.losses import MelSpectralLoss, AdaptiveMelSpectralLoss, TimeFreqResMelSpectralLoss

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_dir", None, "Directory to store checkpoints.")
flags.DEFINE_string("data_dir", None, "Directory of training data (TFRecords).")

def train(data_provider,
          trainer,
          batch_size=32,
          num_steps=1000000,
          steps_per_summary=300,
          steps_per_save=300,
          model_dir='~/tmp/ddsp'):
    """Main training loop."""
    # Get a distributed dataset.
    dataset = data_provider.get_batch(batch_size, shuffle=True, repeats=-1)
    dataset = trainer.distribute_dataset(dataset)
    dataset_iter = iter(dataset)

    # Build model, easiest to just run forward pass.
    trainer.build(next(dataset_iter))

    # Load latest checkpoint if one exists in model_dir.
    trainer.restore(model_dir)

    # Create training loss metrics.
    avg_losses = {name: tf.keras.metrics.Mean(name=name, dtype=tf.float32)
                  for name in trainer.model.loss_names}

    # Set up the summary writer and metrics.
    summary_dir = os.path.join(model_dir, 'summaries', 'train')
    summary_writer = tf.summary.create_file_writer(summary_dir)

    # Save the gin config.
    write_gin_config(summary_writer, model_dir, trainer.step.numpy())

    # Train.
    with summary_writer.as_default():
        for _ in range(num_steps):
            step = trainer.step

            # Take a step.
            losses = trainer.train_step(dataset_iter)

            # Update metrics.
            for k, v in losses.items():
                avg_losses[k].update_state(v)

            # Log the step.
            logging.info('Step:%d Loss:%.2f', step, losses['total_loss'])

            # Write Summaries.
            if step % steps_per_summary == 0:
                for k, metric in avg_losses.items():
                    tf.summary.scalar('losses/{}'.format(k), metric.result(), step=step)
                    metric.reset_states()
                '''for loss in trainer.model.loss_objs:
                    if loss.name == "mel_spectral_loss":
                        for i, fft_layer in enumerate(loss.most_recent_logmags):
                            tf.summary.image("mel_spectrum_%d_target" % i,
                                             fft_layer["target"][:,:,:,tf.newaxis],
                                             step=step,
                                             description=str(fft_layer["meta"]))
                            tf.summary.image("mel_spectrum_%d_value" % i,
                                             fft_layer["value"][:,:,:,tf.newaxis],
                                             step=step,
                                             description=str(fft_layer["meta"]))'''

            # Save Model.
            if step % steps_per_save == 0:
                trainer.save(model_dir)
                summary_writer.flush()

    logging.info('Training Finished!')


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
    #losses = [MelSpectralLoss(sample_rate=data_provider.audio_rate, n_bands=2)]
    #losses = [AdaptiveMelSpectralLoss(sample_rate=data_provider.audio_rate, n_bands=8)]
    losses = [TimeFreqResMelSpectralLoss(sample_rate=data_provider.audio_rate,
                                         time_res=1/data_provider.input_rate)]
    model = ModelBuilder(
        model_type="f0_rnn_fc_hpn_decoder",
        audio_rate=data_provider.audio_rate,
        input_rate=data_provider.input_rate,
        window_secs=data_provider.example_secs,
        f0_denom=n_cylinders,
        checkpoint_dir=FLAGS.checkpoint_dir,
        losses=losses,
        feature_domain="time"
    ).build()

    logging.info("Building trainer...")
    gpus = ["/physical_device:GPU:0", "/physical_device:GPU:1"]
    strategy = get_strategy(gpus=gpus)
    trainer = Trainer(model, strategy)

    logging.info("Initializing training...")
    while True:
        #try:
        train(data_provider, trainer, batch_size=1,
                                          steps_per_summary=1,
                                          steps_per_save=10,
                                          model_dir=FLAGS.checkpoint_dir)
        #except:
        #    logging.info("An error ocurred, reinitializing training from last checkpoint...")

if __name__ == "__main__":
    app.run(main)
