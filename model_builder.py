from data.constants import DEFAULT_WINDOW_SECS, DEFAULT_SAMPLE_RATE, DEFAULT_FRAME_RATE
from models.models import F0RnnFcHPNDecoder

class ModelBuilder:
    '''
    Factory class for building TensorFlow models.
    '''
    @staticmethod
    def create_f0_rnn_fc_hpn_decoder(window_secs=DEFAULT_WINDOW_SECS,
                                     audio_rate=DEFAULT_SAMPLE_RATE,
                                     input_rate=DEFAULT_FRAME_RATE):
        return F0RnnFcHPNDecoder(window_secs=window_secs,
                                 audio_rate=audio_rate,
                                 input_rate=input_rate)