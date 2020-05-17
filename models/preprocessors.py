import tensorflow as tf
import numpy as np
from ddsp.training.preprocessing import Preprocessor, at_least_3d
from ddsp.spectral_ops import F0_RANGE
from ddsp.core import resample, hz_to_midi, oscillator_bank

F0_SUB_RANGE = 1000 # 1000 Mel = 1000 Hz, which is a lot higher than most f0 / n_cyl
MAX_F0_HZ = 138.43 # From the ford_large dataset

def hz_to_mel(f):
    return 1127 * tf.math.log(1 + f/700)

F0_RANGE_MEL = hz_to_mel(MAX_F0_HZ)

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

        # Set additive input
        features["f0_additive"] = features["f0"]

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
        elif self.feature_domain == "osc":
            if features.get("osc", None) is None:
                amplitudes = tf.ones(tf.shape(features["f0"]))
                features["f0_scaled"] = oscillator_bank(self.denom * features["f0"],
                                                        amplitudes,
                                                        sample_rate=self.rate)[:,:,tf.newaxis]
            else:
                features["f0_scaled"] = features["osc"][:,:,tf.newaxis]
        else:
            raise ValueError("%s is not a valid value for feature_domain." % self.feature_domain)

        return features

class OscF0Preprocessor(Preprocessor):
    '''
    Preprocessor for f0 and osc features. Scales f0 envelope by converting to
    MIDI scale and normalizing to [0, 1].
    '''
    def __init__(self, time_steps=None, denom=1., f0_additive="f0", rate=None):
        if time_steps is None:
            raise ValueError("time_steps cannot be None.")
        super().__init__()
        self.time_steps = time_steps
        self.denom = denom
        self.f0_additive = f0_additive
        self.rate = rate

    def __call__(self, features, training=True):
        super().__call__(features, training)
        return self._default_processing(features)

    def _default_processing(self, features):
        '''Always resample to time_steps and scale f0 signal.'''
        # Make sure inputs have the right dimensions, i.e. [batch_size, n_frames, {context dependent}]
        for k in ["f0", "phase", "phase_unwrapped", "osc", "osc_sub",
                  "phase_sub", "phase_unwrapped_sub", "osc_sub_sync",
                  "phase_unwrapped_sub_sync", "phase_sub_sync"]:
            if features.get(k, None) is not None:
                features[k] = at_least_3d(features[k])
                features[k] = resample(features[k], n_timesteps=self.time_steps)
        
        # Divide by denom (e.g. number of cylinders in engine to produce subharmonics)
        features["f0_sub"] = features["f0"] / self.denom

        # Set additive input
        features["f0_additive"] = features[self.f0_additive]
        
        # Generate osc and phase from f0 if missing
        for suffix in ["", "_sub"]:
            if features.get("osc"+suffix, None) is None:
                amplitudes = tf.ones(tf.shape(features["f0"+suffix]))
                features["osc"+suffix] = oscillator_bank(features["f0"+suffix],
                                                amplitudes,
                                                sample_rate=self.rate)[:,:,tf.newaxis]
            if features.get("phase"+suffix, None) is None:
                omegas = 2.0 * np.pi * features["f0"+suffix] / float(self.rate)
                phases = tf.cumsum(omegas, axis=1)
                features["phase_unwrapped"+suffix] = phases
                phases_wrapped = tf.math.mod(phases + np.pi, 2 * np.pi) - np.pi
                features["phase"+suffix] = phases_wrapped

        for prefix in ["osc_sub", "phase_sub", "phase_unwrapped_sub"]:
            if features.get(prefix+"_sync", None) is None:
                features[prefix+"_sync"] = features[prefix]
        
        # Prepare decoder network inputs
        features["f0_scaled"] = hz_to_midi(features["f0"]) / F0_RANGE
        features["f0_scaled_mel"] = hz_to_mel(features["f0"]) / F0_RANGE_MEL
        features["f0_sub_scaled"] = hz_to_mel(features["f0_sub"]) / F0_SUB_RANGE
        for k in ["phase", "phase_sub", "phase_sub_sync"]:
            if features.get(k, None) is not None:
                features[k+"_scaled"] = 0.5 + 0.5 * features[k] / np.pi
        for k in ["osc", "osc_sub", "osc_sub_sync"]:
            if features.get(k, None) is not None:
                features[k+"_scaled"] = 0.5 + 0.5 * features[k]

        return features

class PhaseF0Preprocessor(Preprocessor):
    '''
    Preprocessor for f0 and phase features. Scales f0 envelope by converting to
    Mel scale and normalizing to [0, 1]. Uses phase to determine DCT
    frequencies of the transients.
    '''
    def __init__(self, time_steps=None, denom=1., rate=None):
        if time_steps is None:
            raise ValueError("time_steps cannot be None.")
        super().__init__()
        self.time_steps = time_steps
        self.denom = denom
        self.rate = rate

    def __call__(self, features, training=True):
        super().__call__(features, training)
        return self._default_processing(features)

    def _default_processing(self, features):
        '''Always resample to time_steps and scale input signals.'''
        for k in ["f0", "phase", "phase_unwrapped", "osc", "osc_sub",
                  "phase_sub", "phase_unwrapped_sub", "osc_sub_sync",
                  "phase_unwrapped_sub_sync", "phase_sub_sync"]:
            if features.get(k, None) is not None:
                features[k] = at_least_3d(features[k])
                features[k] = resample(features[k], n_timesteps=self.time_steps)
        
        # Divide by denom (e.g. number of cylinders in engine to produce subharmonics)
        features["f0_sub"] = features["f0"] / self.denom

        # Set additive input
        features["f0_additive"] = features["f0_sub"]
        
        # Prepare decoder network inputs
        features["f0_scaled"] = hz_to_midi(features["f0"]) / F0_RANGE
        features["f0_scaled_mel"] = hz_to_mel(features["f0"]) / F0_RANGE_MEL
        features["f0_sub_scaled"] = hz_to_mel(features["f0_sub"]) / F0_SUB_RANGE
        for k in ["phase", "phase_sub", "phase_sub_sync"]:
            if features.get(k, None) is not None:
                features[k+"_scaled"] = 0.5 + 0.5 * features[k] / np.pi
        for k in ["osc", "osc_sub", "osc_sub_sync"]:
            if features.get(k, None) is not None:
                features[k+"_scaled"] = 0.5 + 0.5 * features[k]

        return features
