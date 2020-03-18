from data.constants import DEFAULT_WINDOW_SECS, DEFAULT_SAMPLE_RATE, DEFAULT_FRAME_RATE
from models.models import F0RnnFcHPNDecoder, OscF0RnnFcHPNDecoder
from models.losses import TimeFreqResMelSpectralLoss

class ModelBuilder:
    '''
    Factory class for building TensorFlow models.
    '''
    def __init__(self, **kwargs):
        self.config = kwargs
        self.model = None
    
    @property
    def model_id(self):
        return self.config.get("model_id", None)
    
    @property
    def data_dir(self):
        return self.config.get("data_dir", None)
    
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
    def feature_domain(self):
        return self.config.get("feature_domain", "freq")
    
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
                                      losses=self.losses,
                                      feature_domain=self.feature_domain)
        elif self.model_type == "osc_f0_rnn_fc_hpn_decoder":
            model = OscF0RnnFcHPNDecoder(window_secs=self.window_secs,
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

def get_model_builder_from_id(model_id):
    '''
    Build a pre-specified model given its ID.
    '''
    if model_id == "200305_1_hpn_ford_mini_freq_time_res_mel_loss":
        return ModelBuilder(
            model_id="200305_1_hpn_ford_mini_freq_time_res_mel_loss",
            data_dir="./data/tfrecord/ford_mini",
            checkpoint_dir="./data/weights/200305_1_hpn_ford_mini_freq_time_res_mel_loss",
            model_type="f0_rnn_fc_hpn_decoder",
            f0_denom=4.0,
            losses=[TimeFreqResMelSpectralLoss(sample_rate=48000, time_res=1/250)],
            feature_domain="time"
        )
    if model_id == "200306_4_hpn_ford_ddsp":
        return ModelBuilder(
            model_id="200306_4_hpn_ford_ddsp",
            data_dir="./data/tfrecord/ford",
            checkpoint_dir="./data/weights/200306_4_hpn_ford_ddsp",
            model_type="f0_rnn_fc_hpn_decoder"
        )
    if model_id == "200306_5_hpn_ford_mel_cyl_time":
        return ModelBuilder(
            model_id="200306_5_hpn_ford_mel_cyl_time",
            data_dir="./data/tfrecord/ford",
            checkpoint_dir="./data/weights/200306_5_hpn_ford_mel_cyl_time",
            model_type="f0_rnn_fc_hpn_decoder",
            f0_denom=4.0,
            losses=[TimeFreqResMelSpectralLoss(sample_rate=48000, time_res=1/250)],
            feature_domain="time"
        )
    if model_id == "200309_1_hpn_ford_mel_cyl_freq":
        return ModelBuilder(
            model_id="200309_1_hpn_ford_mel_cyl_freq",
            data_dir="./data/tfrecord/ford",
            checkpoint_dir="./data/weights/200309_1_hpn_ford_mel_cyl_freq",
            model_type="f0_rnn_fc_hpn_decoder",
            f0_denom=4.0,
            losses=[TimeFreqResMelSpectralLoss(sample_rate=48000, time_res=1/250)],
            feature_domain="freq"
        )
    if model_id == "200310_1_hpn_ford_osc":
        return ModelBuilder(
            model_id="200310_1_hpn_ford_osc",
            data_dir="./data/tfrecord/ford_osc",
            checkpoint_dir="./data/weights/200310_1_hpn_ford_osc",
            model_type="f0_rnn_fc_hpn_decoder",
            f0_denom=4.0,
            losses=[TimeFreqResMelSpectralLoss(sample_rate=48000, time_res=1/250)],
            feature_domain="osc"
        )
    if model_id == "200310_2_hpn_ford_1000fps_osc":
        return ModelBuilder(
            model_id="200310_2_hpn_ford_1000fps_osc",
            data_dir="./data/tfrecord/ford_1000fps_osc",
            checkpoint_dir="./data/weights/200310_2_hpn_ford_1000fps_osc",
            model_type="f0_rnn_fc_hpn_decoder",
            input_rate=1000,
            f0_denom=4.0,
            losses=[TimeFreqResMelSpectralLoss(sample_rate=48000, time_res=1/1000)],
            feature_domain="osc"
        )
    if model_id == "200318_1_hpn_ford_mini_f0_osc":
        return ModelBuilder(
            model_id="200318_1_hpn_ford_mini_f0_osc",
            data_dir="./data/tfrecord/ford_osc_mini",
            checkpoint_dir="./data/weights/200318_1_hpn_ford_mini_f0_osc",
            model_type="osc_f0_rnn_fc_hpn_decoder",
            f0_denom=4.0,
            losses=[TimeFreqResMelSpectralLoss(sample_rate=48000, time_res=1/250)]
        )
    else:
        raise ValueError("%s is not a valid model id." % model_id)
