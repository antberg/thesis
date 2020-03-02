'''
DDSP processor groups representing different signal models.
'''
from ddsp.synths import Additive, FilteredNoise
from ddsp.processors import ProcessorGroup, Add

class HarmonicPlusNoise(ProcessorGroup):
    '''Spectral modeling synthesis (SMS) with harmonic sinusoids.'''
    def __init__(self, window_secs=None, audio_rate=None, input_rate=None, name="harmonic_plus_noise"):
        if window_secs is None:
            raise ValueError("Length of windows (window_secs) must be set.")
        if audio_rate is None:
            raise ValueError("Audio sample rate (audio_rate) must be set.")
        if input_rate is None:
            raise ValueError("Input sample rate (input_rate) must be set.")
        self.window_secs = window_secs
        self.audio_rate = audio_rate
        self.input_rate = input_rate
        self.n_samples = int(window_secs * audio_rate)
        self.additive = Additive(n_samples=self.n_samples, sample_rate=self.audio_rate, name="additive")
        self.subtractive = FilteredNoise(n_samples=self.n_samples, name="subtractive", initial_bias=-2.0)
        self.add = Add()
        dag = [(self.additive, ["amps", "harmonic_distribution", "f0"]),
               (self.subtractive, ["noise_magnitudes"]),
               (self.add, ["additive/signal", "subtractive/signal"])]
        super(HarmonicPlusNoise, self).__init__(dag, name)