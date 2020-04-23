'''
Script for training.
'''
import os
import pprint
import tensorflow as tf
from absl import app, logging, flags
from ddsp.training.train_util import Trainer, write_gin_config, get_strategy

from data.data_provider import TFRecordProvider
from model_builder import ModelBuilder, get_model_builder_from_id
from models.losses import MelSpectralLoss, AdaptiveMelSpectralLoss, TimeFreqResMelSpectralLoss

FLAGS = flags.FLAGS
flags.DEFINE_list("model_ids", [], "IDs of models to train, as defined in ModelConfigs in model_builder.py.")
flags.DEFINE_integer("steps_per_summary", 10, "Training steps per summary.")
flags.DEFINE_integer("steps_per_summary_valid", 100, "Training steps per summary of validation.")
flags.DEFINE_integer("steps_per_save", 5000, "Training steps per checkpoint save.")
flags.DEFINE_integer("n_epochs", 10000, "Number of epochs of training.")
flags.DEFINE_list("devices", None, "Training devices.")

MAX_VALID_BATCH_SIZE = 4#128

def get_valid_losses(model, data_provider):
    batch_size = min(MAX_VALID_BATCH_SIZE, data_provider.n_samples_valid)
    dataset = data_provider.get_batch(batch_size, shuffle=True, repeats=1)
    #dataset = trainer.distribute_dataset(dataset)
    batch = next(iter(dataset))
    _ = model(batch)
    return model.losses_dict

def train(data_provider,
          trainer,
          batch_size=32,
          num_steps=1000000,
          steps_per_summary=10,
          steps_per_summary_valid=10,
          steps_per_save=300,
          model_dir='~/tmp/ddsp',
          data_provider_valid=None):
    """Main training loop."""
    # Get a distributed dataset.
    dataset = data_provider.get_batch(batch_size, shuffle=True, repeats=-1)
    dataset = trainer.distribute_dataset(dataset)
    dataset_iter = iter(dataset)

    # Build model, easiest to just run forward pass.
    trainer.build(next(dataset_iter))

    # Load latest checkpoint if one exists in model_dir.
    trainer.restore(model_dir)

    # Create training and validation loss metrics.
    avg_losses = {name: tf.keras.metrics.Mean(name=name, dtype=tf.float32)
                  for name in trainer.model.loss_names}
    avg_losses_valid = {name: tf.keras.metrics.Mean(name=name, dtype=tf.float32)
                        for name in trainer.model.loss_names}

    # Set up the summary writers and metrics.
    summary_dir = os.path.join(model_dir, 'summaries', 'train')
    summary_writer = tf.summary.create_file_writer(summary_dir)
    summary_dir_valid = os.path.join(model_dir, 'summaries', 'valid')
    summary_writer_valid = tf.summary.create_file_writer(summary_dir_valid)

    # Save the gin config.
    write_gin_config(summary_writer, model_dir, trainer.step.numpy())

    # Train.
    summary_writer.set_as_default()
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
        if step % steps_per_summary == 0 or step == num_steps:
            for k, metric in avg_losses.items():
                tf.summary.scalar('losses/{}'.format(k), metric.result(), step=step)
                metric.reset_states()
            summary_writer.flush()

        # Save Model and evaluate on validation set.
        if step % steps_per_save == 0 or step == num_steps:
            trainer.save(model_dir) 
        
        # Write validation summaries
        if step % steps_per_summary_valid == 0 or step == num_steps:
            losses_valid = get_valid_losses(trainer.model, data_provider_valid)
            for k, v in losses_valid.items():
                avg_losses_valid[k].update_state(v)
            logging.info('Step:%d Validation loss:%.2f', step, losses_valid['total_loss'])
            summary_writer_valid.set_as_default()
            for k, metric in avg_losses_valid.items():
                tf.summary.scalar('losses/{}'.format(k), metric.result(), step=step)
                metric.reset_states()
            summary_writer_valid.flush()
            summary_writer.set_as_default()
        
        # Stop training if specified number of steps has been taken
        if step == num_steps:
            break

    logging.info('Training Finished!')
    return True


def main(argv):
    for model_id in FLAGS.model_ids:
        # Build model
        logging.info("Building model '%s'..." % model_id)
        model_builder = get_model_builder_from_id(model_id)
        checkpoint_dir = model_builder.checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        data_dir = model_builder.data_dir
        batch_size = model_builder.batch_size
        model = model_builder.build()
        config_path = os.path.join(checkpoint_dir, "config.pprint")
        with open(config_path, "w") as f:
            pprint.pprint(model_builder.__dict__, f)

        # Load data
        logging.info("Loading data from '%s'..." % data_dir)
        data_provider = TFRecordProvider(data_dir, split="train")
        data_provider_valid = TFRecordProvider(data_dir, split="valid")
        n_steps = int(data_provider.n_samples_train / batch_size * FLAGS.n_epochs)

        # Build trainer
        logging.info("Building trainer...")
        summary_dir = os.path.join(checkpoint_dir, "summaries", "train")
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        summary_dir_valid = os.path.join(checkpoint_dir, "summaries", "valid")
        if not os.path.exists(summary_dir_valid):
            os.makedirs(summary_dir_valid)
        if FLAGS.devices is not None and len(FLAGS.devices) == 1:
            strategy = tf.distribute.OneDeviceStrategy(FLAGS.devices[0])
        else:
            strategy = tf.distribute.MirroredStrategy(devices=FLAGS.devices)
        trainer = Trainer(model, strategy)

        # Train
        logging.info("Initializing training...")
        logging.info("Will train for %d epochs (%d steps)." % (FLAGS.n_epochs, n_steps))
        done = False
        while not done:
            try:
                done = train(data_provider,
                             trainer,
                             batch_size=batch_size,
                             num_steps=n_steps,
                             steps_per_summary=FLAGS.steps_per_summary,
                             steps_per_summary_valid=FLAGS.steps_per_summary_valid,
                             steps_per_save=FLAGS.steps_per_save,
                             model_dir=checkpoint_dir,
                             data_provider_valid=data_provider_valid)
            except KeyboardInterrupt:
                logging.info("Registered control-C event, stopping.")
                break
            except:
                logging.info("An error ocurred, reinitializing training from last checkpoint...")
        logging.info("Done training '%s'!" % model_id)

if __name__ == "__main__":
    app.run(main)
