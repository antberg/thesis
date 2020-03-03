import tensorflow as tf
from ddsp.training.preprocessing import Preprocessor, at_least_3d
from ddsp.spectral_ops import F0_RANGE
from ddsp.core import resample, hz_to_midi, oscillator_bank

class F0Preprocessor(Preprocessor):
    '''
    Preprocessor for f0 feature. Scales f0 envelope by converting to MIDI scale
    and normalizing to [0, 1].
    '''
    def __init__(self, time_steps=None, denom=1., rate=None, feature_domain="freq"):
        if time_steps is None:
            raise ValueError("time_steps cannot be None.")
        super().__init__()
        self.time_steps = time_steps
        self.denom = denom
        self.rate = rate
        self.feature_domain = feature_domain

    def __call__(self, features, training=True):
        super().__call__(features, training)
        return self._default_processing(features)

    def _default_processing(self, features):
        '''Always resample to time_steps and scale f0 signal.'''
        features["f0"] = at_least_3d(features["f0"])
        features["f0"] = resample(features["f0"], n_timesteps=self.time_steps)
        
        # Divide by denom (e.g. number of cylinders in engine to produce subharmonics)
        features["f0"] /= self.denom

        # Prepare decoder network inputs
        if self.feature_domain == "freq":
            features["f0_scaled"] = hz_to_midi(features["f0"]) / F0_RANGE
        elif self.feature_domain == "freq-old":
            '''DEPRICATED. This option is for backward compability with a version containing a typo.'''
            features["f0_scaled"] = hz_to_midi(self.denom * features["f0"]) / F0_RANGE / self.denom
        elif self.feature_domain == "time":
            amplitudes = tf.ones(tf.shape(features["f0"]))
            features["f0_scaled"] = oscillator_bank(features["f0"],
                                                    amplitudes,
                                                    sample_rate=self.rate)[:,:,tf.newaxis]
        else:
            raise ValueError("%s is not a valid value for feature_domain." % self.feature_domain)

        return features
