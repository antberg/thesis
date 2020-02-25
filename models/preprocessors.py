from ddsp.training.preprocessing import Preprocessor, at_least_3d
from ddsp.spectral_ops import F0_RANGE
from ddsp.core import resample, hz_to_midi

class F0Preprocessor(Preprocessor):
    def __init__(self, time_steps=250):#1000):
        super().__init__()
        self.time_steps = time_steps

    def __call__(self, features, training=True):
        super().__call__(features, training)
        return self._default_processing(features)

    def _default_processing(self, features):
        '''Always resample to time_steps and scale f0 signal.'''
        features["f0"] = at_least_3d(features["f0"])
        features["f0"] = resample(features["f0"], n_timesteps=self.time_steps)
        # For NN training, scale frequency to the range [0, 1].
        features["f0_scaled"] = hz_to_midi(features["f0"]) / F0_RANGE
        return features