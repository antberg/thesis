'''
Evaluate models.
'''
import time
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_datasets as tfds

from .util import Util

class ModelEvaluator:
    '''
    Class for evaluating models.
    '''
    def __init__(self, model_builder, data_provider, loss_function):
        self.model_builder = model_builder
        self.model = model_builder.build()
        self.data_provider = data_provider
        self.loss_function = loss_function
    
    # ================
    # AUDIO GENERATION
    # ================
    def generate_audio(self, batch):
        '''Generate audio for a batch.'''
        features = self.model.encode(batch)
        audio = self.model.decode(features, training=False)
        return audio
    
    def generate_audio_dict_from_example_id(self, example_id, input_keys="all"):
        '''Generate a dictionary with ground-truth audio, synthesized audio etc.'''
        batch = self.data_provider.get_single_batch(batch_size=1, batch_number=example_id)
        return self.generate_audio_dict_from_batch(batch, input_keys)
    
    def generate_audio_dict_from_batch(self, batch, input_keys="all"):
        data_list = list()
        n_batches = batch["f0"].shape[0]
        audio = batch["audio"].numpy() if batch.get("audio", None) is not None else None
        features = self.model.encode(batch)
        audio_synthesized = self.model.decode(features).numpy()
        for batch_id in range(n_batches):
            data = dict()
            data["audio"] = audio[batch_id,:] if audio is not None else None
            data["audio_synthesized"] = audio_synthesized[batch_id,:]
            data["audio_rate"] = self.data_provider.audio_rate
            data["input_rate"] = self.data_provider.input_rate
            data["inputs"] = dict()
            for input_key in self.data_provider.input_keys:
                if input_keys == "all" or input_key in input_keys:
                    data["inputs"][input_key] = features[input_key].numpy()[batch_id,:]
            data_list.append(data)
        return data_list if n_batches > 1 else data_list[0]
    
    # ================
    # INPUT GENERATION
    # ================
    def generate_inputs(self, input_labels, f0_range=(24.24, 138.43), sig=0.25):
        n_inputs = len(input_labels)
        n_samples = self.data_provider.example_secs * self.data_provider.input_rate
        f0_min = f0_range[0]
        f0_max = f0_range[1]
        f0_mid = (f0_min + f0_max) / 2.0
        f0 = np.ndarray((n_inputs, n_samples))
        for i, input_label in enumerate(input_labels):
            if input_label == "const-lo":
                f0[i,:] = f0_min * np.ones(n_samples)
            elif input_label == "const-mid":
                f0[i,:] = f0_mid * np.ones(n_samples)
            elif input_label == "const-hi":
                f0[i,:] = f0_max * np.ones(n_samples)
            elif input_label == "ramp":
                f0[i,:] = np.linspace(f0_min, f0_max, n_samples)
            elif input_label == "osc-fast":
                n_periods = 10
                n = np.linspace(0.0, 2 * np.pi * n_periods, n_samples)
                f0[i,:] = f0_mid + 0.25 * (f0_max - f0_min) * np.sin(n)
            elif input_label == "osc-slow":
                n_periods = 1
                n = np.linspace(0.0, 2 * np.pi * n_periods, n_samples) - np.pi / 2
                f0[i,:] = f0_mid + 0.5 * (f0_max - f0_min) * np.sin(n)
            elif input_label == "outside-lo":
                f0[i,:] = np.linspace(0.1 * f0_min, 0.9 * f0_min, n_samples)
            elif input_label == "outside-hi":
                f0[i,:] = np.linspace(1.1 * f0_max, 2.0 * f0_max, n_samples)
            else:
                raise ValueError("%s is not a valid input label." % input_label)
        if sig > 0.0:
            f0 += np.random.normal(scale=sig, size=(n_inputs, n_samples))
        return f0
    
    def generate_inputs_tensor(self, *args, **kwargs):
        inputs_numpy = self.generate_inputs(*args, **kwargs)
        inputs_tensor = tf.convert_to_tensor(inputs_numpy)
        return {"f0": inputs_tensor}
    
    # ===============
    # FILE MANAGEMENT
    # ===============
    @staticmethod
    def save_audio_to_wav(audio, save_path, sample_rate=48000):
        '''Save audio to wav file.'''
        sf.write(save_path, audio, sample_rate)

    # ==============
    # COMPUTE LOSSES
    # ==============
    def compute_batch_loss(self, batch, inference_time=False):
        '''Compute reconstruction loss for a batch.'''
        target_audio = batch["audio"]
        start = time.clock()
        audio = self.generate_audio(batch)
        elapsed = time.clock() - start
        loss = self.loss_function(audio, target_audio)
        if inference_time:
            return loss, elapsed
        return loss
    
    def compute_total_loss(self, batch_size=32):
        '''Compute total reconstruction loss (and inference time per example) for a dataset.'''
        dataset = self.data_provider.get_batch(batch_size, repeats=1)
        n_samples = self.data_provider.metadata["n_samples_"+self.data_provider.split]
        n_batches = int(n_samples / batch_size)
        if n_batches == 0:
            raise ValueError("batch_size must be smaller than dataset size.")
        batch_loss = np.ndarray(n_batches)
        batch_time = np.ndarray(n_batches)
        for i, batch in enumerate(iter(dataset)):
            print("Computing loss for batch %d (out of %d)..." % (i+1, n_batches))
            batch_loss[i], batch_time[i] = self.compute_batch_loss(batch, inference_time=True)
        data = dict(loss_mean=np.mean(batch_loss), loss_std=np.std(batch_loss),
                    time_mean=np.mean(batch_time), time_std=np.std(batch_time))
        return data

class CQTLoss:
    '''
    Loss based on the constant-Q transform. Based on librosa, so cannot be used in tensorflow.
    '''
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate

    def __call__(self, audio, target_audio):
        loss = 0.0
        n_batches = audio.shape[0]
        for batch_id in range(n_batches):
            cqt_target = Util.get_cqt_spectrogram(target_audio.numpy()[batch_id,:],
                                                  self.sample_rate,
                                                  ref=1.0)
            cqt_synth = Util.get_cqt_spectrogram(audio.numpy()[batch_id,:],
                                                 self.sample_rate,
                                                 ref=1.0)
            loss += np.mean(np.abs(cqt_target - cqt_synth))
        return loss / n_batches
