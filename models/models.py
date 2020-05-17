'''
End-to-end models for neural audio synthesis.
'''
from ddsp.training.models import Autoencoder
from ddsp.losses import SpectralLoss

from .losses import MelSpectralLoss
from .preprocessors import F0Preprocessor, OscF0Preprocessor, PhaseF0Preprocessor
from .decoders import F0RnnFcDecoder, MultiInputRnnFcDecoder
from .processors import HarmonicPlusNoise, HarmonicPlusNoisePlusTransients, HarmonicPlusTransients, HarmonicPlusTransientsPhase

class F0RnnFcHPNDecoder(Autoencoder):
    '''
    Full RNN-FC decoder harmonic-plus-noise synthesizer stack for decoding f0
    signals into audio.
    '''
    def __init__(self, window_secs=None,
                       audio_rate=None,
                       input_rate=None,
                       f0_denom=1.,
                       n_harmonic_distribution=60,
                       n_noise_magnitudes=65,
                       n_rnn=1,
                       losses=None,
                       feature_domain="freq",
                       name="f0_rnn_fc_hpn_decoder"):
        # Initialize preprocessor
        if window_secs * input_rate % 1.0 != 0.0:
            raise ValueError("window_secs and input_rate must result in an integer number of samples per window.")
        time_steps = int(window_secs * input_rate)
        preprocessor = F0Preprocessor(time_steps=time_steps,
                                      denom=f0_denom,
                                      rate=input_rate,
                                      feature_domain=feature_domain)

        # Initialize decoder
        decoder = F0RnnFcDecoder(rnn_channels=512,
                                 rnn_type="gru",
                                 n_rnn=n_rnn,
                                 ch=512,
                                 layers_per_stack=3,
                                 output_splits=(("amps", 1),
                                                ("harmonic_distribution", n_harmonic_distribution),
                                                ("noise_magnitudes", n_noise_magnitudes)))
        
        # Initialize processor group
        processor_group = HarmonicPlusNoise(window_secs=window_secs,
                                            audio_rate=audio_rate,
                                            input_rate=input_rate)
        
        # Initialize losses
        if losses is None:
            losses = [SpectralLoss(fft_sizes=(8192, 4096, 2048, 1024, 512, 256, 128),
                                   loss_type="L1",
                                   mag_weight=1.0,
                                   logmag_weight=1.0)]
        
        # Call parent constructor
        super(F0RnnFcHPNDecoder, self).__init__(preprocessor=preprocessor,
                                                encoder=None,
                                                decoder=decoder,
                                                processor_group=processor_group,
                                                losses=losses)

class OscF0RnnFcHPNDecoder(Autoencoder):
    '''
    Full RNN-FC decoder harmonic-plus-noise synthesizer stack for decoding f0
    and osc signals into audio.
    '''
    def __init__(self, window_secs=None,
                       audio_rate=None,
                       input_rate=None,
                       f0_denom=1.,
                       n_harmonic_distribution=60,
                       n_noise_magnitudes=65,
                       rnn_channels=512,
                       layers_per_stack=3,
                       input_keys=["f0_sub_scaled", "osc_scaled"],
                       f0_additive="f0",
                       losses=None,
                       name="osc_f0_rnn_fc_hpn_decoder"):
        # Initialize preprocessor
        if window_secs * input_rate % 1.0 != 0.0:
            raise ValueError("window_secs and input_rate must result in an integer number of samples per window.")
        time_steps = int(window_secs * input_rate)
        preprocessor = OscF0Preprocessor(time_steps=time_steps,
                                         denom=f0_denom,
                                         f0_additive=f0_additive,
                                         rate=input_rate)

        # Initialize decoder
        decoder = MultiInputRnnFcDecoder(rnn_channels=rnn_channels,
                                         rnn_type="gru",
                                         ch=512,
                                         layers_per_stack=layers_per_stack,
                                         input_keys=input_keys,
                                         output_splits=(("amps", 1),
                                                        ("harmonic_distribution", n_harmonic_distribution),
                                                        ("noise_magnitudes", n_noise_magnitudes)))
        
        # Initialize processor group
        processor_group = HarmonicPlusNoise(window_secs=window_secs,
                                            audio_rate=audio_rate,
                                            input_rate=input_rate)
        
        # Initialize losses
        if losses is None:
            losses = [SpectralLoss(fft_sizes=(8192, 4096, 2048, 1024, 512, 256, 128),
                                   loss_type="L1",
                                   mag_weight=1.0,
                                   logmag_weight=1.0)]
        
        # Call parent constructor
        super(OscF0RnnFcHPNDecoder, self).__init__(preprocessor=preprocessor,
                                                   encoder=None,
                                                   decoder=decoder,
                                                   processor_group=processor_group,
                                                   losses=losses)

class OscF0RnnFcHPNTDecoder(Autoencoder):
    '''
    Full RNN-FC decoder harmonic-plus-noise-plus-transients synthesizer stack
    for decoding f0 and osc signals into audio.
    '''
    def __init__(self, window_secs=None,
                       audio_rate=None,
                       input_rate=None,
                       f0_denom=1.,
                       n_harmonic_distribution=60,
                       n_noise_magnitudes=65,
                       n_transient_distribution=200,
                       rnn_channels=512,
                       layers_per_stack=3,
                       input_keys=["f0_sub_scaled", "osc_scaled"],
                       f0_additive="f0",
                       losses=None,
                       name="multi_input_rnn_fc_hpnt_decoder"):
        # Initialize preprocessor
        if window_secs * input_rate % 1.0 != 0.0:
            raise ValueError("window_secs and input_rate must result in an integer number of samples per window.")
        time_steps = int(window_secs * input_rate)
        preprocessor = OscF0Preprocessor(time_steps=time_steps,
                                         denom=f0_denom,
                                         f0_additive=f0_additive,
                                         rate=input_rate)

        # Initialize decoder
        decoder = MultiInputRnnFcDecoder(rnn_channels=rnn_channels,
                                         rnn_type="gru",
                                         ch=512,
                                         layers_per_stack=layers_per_stack,
                                         input_keys=input_keys,
                                         output_splits=(("amps", 1),
                                                        ("harmonic_distribution", n_harmonic_distribution),
                                                        ("noise_magnitudes", n_noise_magnitudes),
                                                        ("transient_amps", 1),
                                                        ("transient_distribution", n_transient_distribution)))
        
        # Initialize processor group
        processor_group = HarmonicPlusNoisePlusTransients(window_secs=window_secs,
                                                          audio_rate=audio_rate,
                                                          input_rate=input_rate)
        
        # Initialize losses
        if losses is None:
            losses = [SpectralLoss(fft_sizes=(8192, 4096, 2048, 1024, 512, 256, 128),
                                   loss_type="L1",
                                   mag_weight=1.0,
                                   logmag_weight=1.0)]
        
        # Call parent constructor
        super(OscF0RnnFcHPNTDecoder, self).__init__(preprocessor=preprocessor,
                                                    encoder=None,
                                                    decoder=decoder,
                                                    processor_group=processor_group,
                                                    losses=losses)

class OscF0RnnFcHPTDecoder(Autoencoder):
    '''
    Full RNN-FC decoder harmonic-plus-transients synthesizer stack
    for decoding f0 and osc signals into audio.
    '''
    def __init__(self, window_secs=None,
                       audio_rate=None,
                       input_rate=None,
                       f0_denom=1.,
                       n_harmonic_distribution=60,
                       n_transient_distribution=200,
                       rnn_channels=512,
                       layers_per_stack=3,
                       input_keys=["f0_sub_scaled", "osc_scaled"],
                       f0_additive="f0",
                       losses=None,
                       name="multi_input_rnn_fc_hpt_decoder"):
        # Initialize preprocessor
        if window_secs * input_rate % 1.0 != 0.0:
            raise ValueError("window_secs and input_rate must result in an integer number of samples per window.")
        time_steps = int(window_secs * input_rate)
        preprocessor = OscF0Preprocessor(time_steps=time_steps,
                                         denom=f0_denom,
                                         f0_additive=f0_additive,
                                         rate=input_rate)

        # Initialize decoder
        decoder = MultiInputRnnFcDecoder(rnn_channels=rnn_channels,
                                         rnn_type="gru",
                                         ch=512,
                                         layers_per_stack=layers_per_stack,
                                         input_keys=input_keys,
                                         output_splits=(("amps", 1),
                                                        ("harmonic_distribution", n_harmonic_distribution),
                                                        ("transient_amps", 1),
                                                        ("transient_distribution", n_transient_distribution)))
        
        # Initialize processor group
        processor_group = HarmonicPlusTransients(window_secs=window_secs,
                                                 audio_rate=audio_rate,
                                                 input_rate=input_rate)
        
        # Initialize losses
        if losses is None:
            losses = [SpectralLoss(fft_sizes=(8192, 4096, 2048, 1024, 512, 256, 128),
                                   loss_type="L1",
                                   mag_weight=1.0,
                                   logmag_weight=1.0)]
        
        # Call parent constructor
        super(OscF0RnnFcHPTDecoder, self).__init__(preprocessor=preprocessor,
                                                   encoder=None,
                                                   decoder=decoder,
                                                   processor_group=processor_group,
                                                   losses=losses)

class PhaseF0RnnFcHPTDecoder(Autoencoder):
    '''
    Full RNN-FC decoder harmonic-plus-transients-phase synthesizer stack
    for decoding f0 signals into audio.
    '''
    def __init__(self, window_secs=None,
                       audio_rate=None,
                       input_rate=None,
                       f0_denom=1.,
                       n_harmonic_distribution=60,
                       n_transient_distribution=200,
                       n_transients_per_period=1,
                       initial_phase_shift=0.0,
                       rnn_channels=512,
                       layers_per_stack=3,
                       input_keys=["f0_sub_scaled"],
                       losses=None,
                       name="multi_input_rnn_fc_hpt_phase_decoder"):
        # Initialize preprocessor
        if window_secs * input_rate % 1.0 != 0.0:
            raise ValueError("window_secs and input_rate must result in an integer number of samples per window.")
        time_steps = int(window_secs * input_rate)
        preprocessor = PhaseF0Preprocessor(time_steps=time_steps,
                                           denom=f0_denom,
                                           rate=input_rate)

        # Initialize decoder
        decoder = MultiInputRnnFcDecoder(rnn_channels=rnn_channels,
                                         rnn_type="gru",
                                         ch=512,
                                         layers_per_stack=layers_per_stack,
                                         input_keys=input_keys,
                                         output_splits=(("amps", 1),
                                                        ("harmonic_distribution", n_harmonic_distribution),
                                                        ("transient_amps", 1),
                                                        ("transient_distribution", n_transient_distribution)))
        
        # Initialize processor group
        processor_group = HarmonicPlusTransientsPhase(window_secs=window_secs,
                                                      audio_rate=audio_rate,
                                                      input_rate=input_rate,
                                                      n_transients_per_period=n_transients_per_period,
                                                      initial_phase_shift=initial_phase_shift)
        
        # Initialize losses
        if losses is None:
            losses = [SpectralLoss(fft_sizes=(8192, 4096, 2048, 1024, 512, 256, 128),
                                   loss_type="L1",
                                   mag_weight=1.0,
                                   logmag_weight=1.0)]
        
        # Call parent constructor
        super(PhaseF0RnnFcHPTDecoder, self).__init__(preprocessor=preprocessor,
                                                     encoder=None,
                                                     decoder=decoder,
                                                     processor_group=processor_group,
                                                     losses=losses)
