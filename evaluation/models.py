'''
Evaluate models.
'''
import numpy as np
import soundfile as sf

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
        batch = self.data_provider.get_single_batch(batch_number=example_id)
        data = dict()
        data["audio"] = batch["audio"].numpy().flatten()
        data["audio_synthesized"] = self.generate_audio(batch).numpy().flatten()
        data["audio_rate"] = self.data_provider.audio_rate
        data["input_rate"] = self.data_provider.input_rate
        data["inputs"] = dict()
        for input_key in self.data_provider.input_keys:
            if input_keys == "all" or input_key in input_keys:
                data["inputs"][input_key] = batch[input_key].numpy().flatten()
        return data
    
    # ================
    # INPUT GENERATION
    # ================
    def generate_input(self, input_label=None):
        if input_label is None:
            raise ValueError("input_label must be given.")
        raise NotImplementedError
    
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
    def compute_batch_loss(self, batch):
        '''Compute reconstruction loss for a batch.'''
        target_audio = batch["audio"]
        audio = self.generate_audio(batch)
        loss = self.loss_function(audio, target_audio)
        return loss
    
    def compute_total_loss(self, batch_size=32):
        '''Compute total reconstruction loss for a dataset.'''
        dataset = self.data_provider.get_batch(batch_size, repeats=1)
        n_samples = self.data_provider.metadata["n_samples_"+self.data_provider.split]
        n_batches = int(n_samples / batch_size)
        if n_batches == 0:
            raise ValueError("batch_size must be smaller than dataset size.")
        batch_loss = np.ndarray(n_batches)
        for i, batch in enumerate(iter(dataset)):
            if i % 1 == 0:
                print("Computing loss for batch %d (out of %d)..." % (i+1, n_batches))
            batch_loss[i] = self.compute_batch_loss(batch)
        return np.mean(batch_loss), np.std(batch_loss)

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
                                                  scale="amp")
            cqt_synth = Util.get_cqt_spectrogram(audio.numpy()[batch_id,:],
                                                 self.sample_rate,
                                                 scale="amp")
            loss += np.mean(np.abs(cqt_target - cqt_synth))
        return loss / n_batches
