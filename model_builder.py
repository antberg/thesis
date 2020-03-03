from data.constants import DEFAULT_WINDOW_SECS, DEFAULT_SAMPLE_RATE, DEFAULT_FRAME_RATE
from models.models import F0RnnFcHPNDecoder

class ModelBuilder:
    '''
    Factory class for building TensorFlow models.
    '''
    def __init__(self, **kwargs):
        self.config = kwargs
        self.model = None
    
    @property
    def checkpoint_dir(self):
        return self.config.get("checkpoint_dir", ".")
    
    @property
    def window_secs(self):
        return self.config.get("window_secs", DEFAULT_WINDOW_SECS)
    
    @property
    def audio_rate(self):
        return self.config.get("audio_rate", DEFAULT_SAMPLE_RATE)
    
    @property
    def input_rate(self):
        return self.config.get("input_rate", DEFAULT_FRAME_RATE)
    
    @property
    def f0_denom(self):
        return self.config.get("f0_denom", 1.)
    
    @property
    def n_harmonic_distribution(self):
        return self.config.get("n_harmonic_distribution", 60)
    
    @property
    def n_noise_magnitudes(self):
        return self.config.get("n_noise_magnitudes", 65)
    
    @property
    def losses(self):
        return self.config.get("losses", None)
    
    @property
    def model_type(self):
        return self.config.get("model_type", None)
    
    def _get_model(self):
        if not self.model_type:
            raise ValueError("model_type must be set.")
        if self.model_type == "f0_rnn_fc_hpn_decoder":
            model = F0RnnFcHPNDecoder(window_secs=self.window_secs,
                                      audio_rate=self.audio_rate,
                                      input_rate=self.input_rate,
                                      f0_denom=self.f0_denom,
                                      n_harmonic_distribution=self.n_harmonic_distribution,
                                      n_noise_magnitudes=self.n_noise_magnitudes,
                                      losses=self.losses)
        else:
            raise ValueError("%s is not a valid model_type." % self.model_type)

        return model
    
    def build(self):
        self.model = self._get_model()
        if self.checkpoint_dir:
            self.model.restore(self.checkpoint_dir)
        
        return self.model