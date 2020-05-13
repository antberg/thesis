import os
from absl import app, logging, flags
import pprint
import numpy as np

from models.losses import TimeFreqResMelSpectralLoss
from data.data_provider import TFRecordProvider
from evaluation.models import ModelEvaluator, CQTLoss
from evaluation.util import Util
from model_builder import get_model_builder_from_id

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_list("model_ids", None,
                  "Models to evaluate")
flags.DEFINE_bool("show", False,
                  "Whether to show the plots.")
flags.DEFINE_bool("summary", False,
                  "Whether to print model summary.")
flags.DEFINE_bool("losses", False,
                  "Whether to compute reconstruction losses.")
flags.DEFINE_string("loss_type", "mel",
                    "Type of loss to compute.")
flags.DEFINE_bool("reconstruct", False,
                  "Whether to reconstruct audio of given examples.")
flags.DEFINE_string("spec_type", "mel",
                    "Type of spectrograms to plot ('mel' or 'cqt').")
flags.DEFINE_list("ex_ids_train", [1, 127, 78],
                  "Ids of examples to reconstruct from train split (if --reconstruct).")
flags.DEFINE_list("ex_ids_test", [1, 42, 49],
                  "Ids of examples to reconstruct from test split (if --reconstruct).")
flags.DEFINE_bool("control", False,
                  "Whether to synthesize audio based on specified control signals.")
flags.DEFINE_list("control_signals", ["const-lo", "const-mid", "const-hi", "ramp",
                                      "osc-fast", "osc-slow", "outside-lo", "outside-hi"],
                  "Synthesized control signals to use (if --control).")

# Define global constants
FIG_BASE_DIR = os.path.join(".", "evaluation", "fig", "models")
FIG_EXTENSION = "pdf"
LOSS_TYPES = ["mel", "cqt"]

def main(argv):
    # Check preconditions
    if FLAGS.model_ids is None:
        raise ValueError("model_ids must be specified.")
    if FLAGS.loss_type not in LOSS_TYPES:
        raise ValueError("loss_type must be one in %s." % str(LOSS_TYPES))
    if FLAGS.reconstruct:
        if FLAGS.ex_ids_train is None:
            raise ValueError("ex_ids_train must be set if reconstruct is True.")
        FLAGS.ex_ids_train = [int(i) for i in FLAGS.ex_ids_train]
        if FLAGS.ex_ids_test is None:
            raise ValueError("ex_ids_test must be set if reconstruct is True.")
        FLAGS.ex_ids_test = [int(i) for i in FLAGS.ex_ids_test]

    # Initialize loss
    if FLAGS.loss_type == "mel":
        loss_funcion = TimeFreqResMelSpectralLoss(sample_rate=48000,
                                                  time_res=1/250,
                                                  loss_type="L1",
                                                  mag_weight=0.0,
                                                  logmag_weight=1.0)
    elif FLAGS.loss_type == "cqt":
        loss_funcion = CQTLoss(sample_rate=48000)

    # Evaluate each specified model
    for model_id in FLAGS.model_ids:
        fig_dir = os.path.join(FIG_BASE_DIR, model_id)
        data_dir = os.path.join(fig_dir, "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Initialize model builder
        model_builder = get_model_builder_from_id(model_id)

        # Evaluate both training and test data
        for split in ["train", "test"]:
            logging.info("Evaluating %s on %s data..." % (model_id, split))

            # Initialize data provider and model evaluator
            data_provider = TFRecordProvider(model_builder.data_dir, split=split)
            summary = FLAGS.summary and split == "train"
            evaluator = ModelEvaluator(model_builder, data_provider, loss_funcion, summary=summary)

            # Compute loss and inference times
            if FLAGS.losses:
                logging.info("Computing %s loss and inference time for %s..." % (split, model_id))
                data_path = os.path.join(data_dir, "loss_%s_%s.pprint" % (FLAGS.loss_type, split))
                data = evaluator.compute_total_loss(batch_size=1)
                logging.info("%s loss: %.5f +- %.5f" % (FLAGS.loss_type, data["loss_mean"], data["loss_std"]))
                logging.info("mean inference time per example: %.5f +- %.5f" % (data["time_mean"], data["time_std"]))
                with open(data_path, "w") as data_file:
                    pprint.pprint(data, data_file)
                logging.info("Saved loss and inference time in '%s'." % data_path)
            
            # Reconstruct specified examples and store audio and spectrograms
            if FLAGS.reconstruct:
                for example_id in FLAGS["ex_ids_"+split]._value:
                    logging.info("Generating audio and spectrograms from %s example %d for %s..." % (split, example_id, model_id))
                    
                    # Generate audio
                    data = evaluator.generate_audio_dict_from_example_id(example_id)

                    # Store audio
                    audio_rec_path = os.path.join(data_dir, "audio_%s_%d_rec.wav" % (split, example_id))
                    audio_syn_path = os.path.join(data_dir, "audio_%s_%d_syn.wav" % (split, example_id))
                    evaluator.save_audio_to_wav(data["audio"], audio_rec_path, data["audio_rate"])
                    evaluator.save_audio_to_wav(data["audio_synthesized"], audio_syn_path, data["audio_rate"])

                    # Store spectrograms
                    spec_rec_path = os.path.join(fig_dir, "spec_%s_%d_rec.%s" % (split, example_id, FIG_EXTENSION))
                    spec_syn_path = os.path.join(fig_dir, "spec_%s_%d_syn.%s" % (split, example_id, FIG_EXTENSION))
                    spec_diff_path = os.path.join(fig_dir, "spec_%s_%d_diff.%s" % (split, example_id, FIG_EXTENSION))
                    Util.plot_spectrogram_from_dict(data, audio_key="audio",
                                                          spec_type=FLAGS.spec_type,
                                                          save_path=spec_rec_path,
                                                          show=FLAGS.show)
                    Util.plot_spectrogram_from_dict(data, audio_key="audio_synthesized",
                                                          spec_type=FLAGS.spec_type,
                                                          save_path=spec_syn_path,
                                                          show=FLAGS.show)
                    data["audio_diff"] = np.abs(data["audio"] - data["audio_synthesized"])
                    Util.plot_spectrogram_from_dict(data, audio_key="audio_diff",
                                                          spec_type=FLAGS.spec_type,
                                                          save_path=spec_diff_path,
                                                          show=FLAGS.show)

        # Synthesize audio based on specified control signals
        if FLAGS.control:
            logging.info("Generating audio based on synthesized control signals...")
            input_tensor = evaluator.generate_inputs_tensor(FLAGS.control_signals)
            data_list = evaluator.generate_audio_dict_from_batch(input_tensor)
            if not isinstance(data_list, list): # make sure data_list is a list (and not a dict)
                data_list = [data_list]
            for data, control_signal_label in zip(data_list, FLAGS.control_signals):
                logging.info("Saving audio, input plots and spectrogram for %s..." % control_signal_label)

                # Store audio
                audio_path = os.path.join(data_dir, "control_%s.wav" % control_signal_label)
                evaluator.save_audio_to_wav(data["audio_synthesized"], audio_path, data["audio_rate"])

                # Plot inputs
                fig_path = os.path.join(fig_dir, "inputs_%s.%s" % (control_signal_label, FIG_EXTENSION))
                Util.plot_inputs_from_dict(data, plot_audio=False,
                                                 input_keys=["f0", "phase_sub"],
                                                 save_path=fig_path,
                                                 show=FLAGS.show)

                # Plot spectrogram
                spec_path = os.path.join(fig_dir, "control_%s.%s" % (control_signal_label, FIG_EXTENSION))
                Util.plot_spectrogram_from_dict(data, audio_key="audio_synthesized",
                                                      spec_type=FLAGS.spec_type,
                                                      save_path=spec_path,
                                                      show=FLAGS.show)

if __name__ == "__main__":
    app.run(main)
