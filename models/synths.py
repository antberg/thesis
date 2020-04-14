import numpy as np
import tensorflow as tf
from ddsp import core, processors

class AdditiveLinspace(processors.Processor):
    '''
    Synthesize audio with a bank of sinusoidal oscillators with linearly
    spaced frequencies.
    '''
    def __init__(self,
                 n_samples=64000,
                 sample_rate=16000,
                 f_interval=None,
                 scale_fn=core.exp_sigmoid,
                 name="additive_linear"):
        super().__init__(name=name)
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.f_interval = (0.0, sample_rate/2) if f_interval is None else f_interval
        if self.f_interval[1] > self.sample_rate/2:
            raise ValueError("Frequency interval must be below Nyquist.")
        self.scale_fn = scale_fn

    def get_controls(self,
                     amplitudes,
                     frequency_distribution):
        '''Convert network output tensors into a dictionary of synthesizer controls.

        Args:
            amplitudes: 3-D Tensor of synthesizer controls, of shape
                [batch, time, 1].
            frequency_distribution: 3-D Tensor of synthesizer controls, of shape
                [batch, time, n_frequencies].

        Returns:
            controls: Dictionary of tensors of synthesizer controls.
        '''
        # Scale the amplitudes.
        if self.scale_fn is not None:
            amplitudes = self.scale_fn(amplitudes)
            frequency_distribution = self.scale_fn(frequency_distribution)

        # Normalize
        frequency_distribution /= tf.reduce_sum(frequency_distribution,
                                                axis=-1,
                                                keepdims=True)

        return {"amplitudes": amplitudes,
                "frequency_distribution": frequency_distribution}

    def get_signal(self, amplitudes, frequency_distribution):
        '''Synthesize audio with additive synthesizer from controls.

        Args:
            amplitudes: Amplitude tensor of shape [batch, n_frames, 1]. Expects
                float32 that is strictly positive.
            frequency_distribution: Tensor of shape [batch, n_frames, n_harmonics].
                Expects float32 that is strictly positive and normalized in the last
                dimension.

        Returns:
            signal: A tensor of harmonic waves of shape [batch, n_samples].
        '''
        signal = self.additive_synthesis(
                amplitudes=amplitudes,
                frequency_distribution=frequency_distribution,
                n_samples=self.n_samples,
                sample_rate=self.sample_rate)
        return signal
    
    def additive_synthesis(self,
                           amplitudes,
                           frequency_shifts=None,
                           frequency_distribution=None,
                           n_samples=64000,
                           sample_rate=16000,
                           amp_resample_method="window"):
        '''Generate audio from frame-wise monophonic harmonic oscillator bank.

        Args:
            amplitudes: Frame-wise oscillator peak amplitude. Shape [batch_size,
                n_frames, 1].
            frequency_shifts: Harmonic frequency variations (Hz), zero-centered. Total
                frequency of a harmonic is equal to (frequencies * (1 +
                frequency_shifts)). Shape [batch_size, n_frames, n_harmonics].
            frequency_distribution: Harmonic amplitude variations, ranged zero to one.
                Total amplitude of a harmonic is equal to (amplitudes *
                frequency_distribution). Shape [batch_size, n_frames, n_harmonics].
            n_samples: Total length of output audio. Interpolates and crops to this.
            sample_rate: Sample rate.
            amp_resample_method: Mode with which to resample amplitude envelopes.

        Returns:
            audio: Output audio. Shape [batch_size, n_samples, 1]
        '''
        amplitudes = core.tf_float32(amplitudes)
        batch_size = amplitudes.shape[0]
        n_frames = amplitudes.shape[1]

        if frequency_distribution is not None:
            frequency_distribution = core.tf_float32(frequency_distribution)
            n_frequencies = int(frequency_distribution.shape[-1])
        elif harmonic_shifts is not None:
            harmonic_shifts = core.tf_float32(harmonic_shifts)
            n_frequencies = int(frequency_shifts.shape[-1])
        else:
            n_frequencies = 1

        # Create frequencies [batch_size, n_frames, n_frequencies].
        frequencies = self.get_linear_frequencies(batch_size, n_frames, n_frequencies)
        if frequency_shifts is not None:
            frequencies *= (1.0 + harmonic_shifts)

        # Create harmonic amplitudes [batch_size, n_frames, n_frequencies].
        if frequency_distribution is not None:
            frequency_amplitudes = amplitudes * frequency_distribution
        else:
            frequency_amplitudes = amplitudes

        # Create sample-wise envelopes.
        frequency_envelopes = core.resample(frequencies, n_samples)  # cycles/sec
        amplitude_envelopes = core.resample(frequency_amplitudes,
                                            n_samples,
                                            method=amp_resample_method)

        # Synthesize from harmonics [batch_size, n_samples].
        audio = core.oscillator_bank(frequency_envelopes,
                                     amplitude_envelopes,
                                     sample_rate=sample_rate)
        return audio
    
    def get_linear_frequencies(self, batch_size, n_frames, n_frequencies):
        '''Get linearly spaced frequencies.'''
        # Create linearly spaced frequencies.
        d_f = (self.f_interval[1] - self.f_interval[0]) / (n_frequencies + 2)
        frequencies = tf.linspace(self.f_interval[0] + d_f,
                                  self.f_interval[1] - d_f,
                                  n_frequencies)[tf.newaxis,tf.newaxis,:]

        # Repeat across batch and frame dimensions.
        frequencies = tf.repeat(frequencies, repeats=n_frames, axis=1)
        frequencies = tf.repeat(frequencies, repeats=batch_size, axis=0)
        return frequencies

class Transients(AdditiveLinspace):
    '''
    Synthesize audio transients spaced linearly in time.
    '''
    def __init__(self,
                 n_samples=64000,
                 sample_rate=16000,
                 f_interval=None,
                 scale_fn=core.exp_sigmoid,
                 name="transients"):
        super().__init__(n_samples, sample_rate, f_interval, scale_fn, name)
    
    def get_signal(self, *args, **kwargs):
        signal = super().get_signal(*args, **kwargs)
        return tf.signal.idct(signal)

class AdditiveDCT(processors.Processor):
    '''
    Synthesize signals with a bank of sinusoidal oscillators with frequencies
    in the DCT domain spaced by an instantaneous (unwrapped) phase signal.
    '''
    def __init__(self,
                 n_samples=64000,
                 sample_rate=16000,
                 n_transients_per_period=1,
                 initial_phase_shift=0.0,
                 scale_fn=core.exp_sigmoid,
                 name="additive_dct"):
        super().__init__(name=name)
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.n_transients_per_period = n_transients_per_period
        self.initial_phase_shift = initial_phase_shift
        self.scale_fn = scale_fn

    def get_controls(self,
                     amplitudes,
                     frequency_distribution,
                     phase):
        '''Convert network output tensors into a dictionary of synthesizer controls.

        Args:
            amplitudes: 3-D Tensor of synthesizer controls, of shape
                [batch, time, 1].
            frequency_distribution: 3-D Tensor of synthesizer controls, of shape
                [batch, time, n_frequencies].
            phase: Unwrapped phase, of shape [batch, time, 1].

        Returns:
            controls: Dictionary of tensors of synthesizer controls.
        '''
        # Scale the amplitudes.
        if self.scale_fn is not None:
            amplitudes = self.scale_fn(amplitudes)
            frequency_distribution = self.scale_fn(frequency_distribution)

        # Bandlimit the DCT frequency distribution.
        n_frequencies = int(frequency_distribution.shape[-1])
        frequencies = self.get_dct_frequencies(phase, n_frequencies)
        frequency_distribution = core.remove_above_nyquist(frequencies,
                                                           frequency_distribution,
                                                           self.sample_rate)

        # Normalize
        frequency_distribution /= tf.reduce_sum(frequency_distribution,
                                                axis=-1,
                                                keepdims=True)

        return {"amplitudes": amplitudes,
                "frequency_distribution": frequency_distribution,
                "phase": phase}

    def get_signal(self,
                   amplitudes,
                   frequency_distribution,
                   phase):
        '''Synthesize audio with additive synthesizer from controls.

        Args:
            amplitudes: Amplitude tensor of shape [batch, n_frames, 1]. Expects
                float32 that is strictly positive.
            frequency_distribution: Tensor of shape [batch, n_frames, n_harmonics].
                Expects float32 that is strictly positive and normalized in the last
                dimension.
            phase: Unwrapped phase, of shape [batch, time, 1].

        Returns:
            signal: A tensor of harmonic waves of shape [batch, n_samples].
        '''
        signal = self.additive_synthesis(
            phase=phase,
            amplitudes=amplitudes,
            frequency_distribution=frequency_distribution,
            n_samples=self.n_samples,
            sample_rate=self.sample_rate)
        return signal
    
    def additive_synthesis(self,
                           phase,
                           amplitudes,
                           frequency_shifts=None,
                           frequency_distribution=None,
                           n_samples=64000,
                           sample_rate=16000,
                           amp_resample_method="window"):
        '''Generate audio from frame-wise monophonic harmonic oscillator bank.

        Args:
            phase: Frame-wise instantaneous phase. Shape [batch_size, n_frames, 1].
            amplitudes: Frame-wise oscillator peak amplitude. Shape [batch_size,
                n_frames, 1].
            frequency_shifts: Harmonic frequency variations (Hz), zero-centered. Total
                frequency of a harmonic is equal to (frequencies * (1 +
                frequency_shifts)). Shape [batch_size, n_frames, n_harmonics].
            frequency_distribution: Harmonic amplitude variations, ranged zero to one.
                Total amplitude of a harmonic is equal to (amplitudes *
                frequency_distribution). Shape [batch_size, n_frames, n_harmonics].
            n_samples: Total length of output audio. Interpolates and crops to this.
            sample_rate: Sample rate.
            amp_resample_method: Mode with which to resample amplitude envelopes.

        Returns:
            audio: Output audio. Shape [batch_size, n_samples, 1]
        '''
        phase = core.tf_float32(phase)
        amplitudes = core.tf_float32(amplitudes)

        if frequency_distribution is not None:
            frequency_distribution = core.tf_float32(frequency_distribution)
            n_frequencies = int(frequency_distribution.shape[-1])
        elif harmonic_shifts is not None:
            harmonic_shifts = core.tf_float32(harmonic_shifts)
            n_frequencies = int(frequency_shifts.shape[-1])
        else:
            n_frequencies = 1

        # Create frequencies from phase signal [batch_size, n_frames, n_frequencies].
        frequencies = self.get_dct_frequencies(phase, n_frequencies)
        #frequencies = self.get_linear_frequencies(phase.shape[0], phase.shape[1], n_frequencies)
        if frequency_shifts is not None:
            frequencies *= (1.0 + harmonic_shifts)

        # Create harmonic amplitudes [batch_size, n_frames, n_frequencies].
        if frequency_distribution is not None:
            frequency_amplitudes = amplitudes * frequency_distribution
        else:
            frequency_amplitudes = amplitudes

        # Create sample-wise envelopes.
        frequency_envelopes = core.resample(frequencies, n_samples)  # cycles/sec
        amplitude_envelopes = core.resample(frequency_amplitudes,
                                            n_samples,
                                            method=amp_resample_method)

        # Synthesize from harmonics [batch_size, n_samples].
        audio = core.oscillator_bank(frequency_envelopes,
                                     amplitude_envelopes,
                                     sample_rate=sample_rate)
        return audio
    
    # NOTE: TEMPORARY METHOD FOR TROUBLE SHOOTING
    def get_linear_frequencies(self, batch_size, n_frames, n_frequencies):
        '''Get linearly spaced frequencies.'''
        # Create linearly spaced frequencies.
        d_f = (self.f_interval[1] - self.f_interval[0]) / (n_frequencies + 2)
        frequencies = tf.linspace(self.f_interval[0] + d_f,
                                  self.f_interval[1] - d_f,
                                  n_frequencies)[tf.newaxis,tf.newaxis,:]

        # Repeat across batch and frame dimensions.
        frequencies = tf.repeat(frequencies, repeats=n_frames, axis=1)
        frequencies = tf.repeat(frequencies, repeats=batch_size, axis=0)
        return frequencies
    
    def get_dct_frequencies(self, phase, n_frequencies):
        '''Get DCT frequencies from instantaneous (unwrapped) phase.'''
        # Create grid of DCT frequencies corresponding to each transient.
        batch_size = phase.shape[0]
        n_frames = phase.shape[1]
        nyquist = self.sample_rate / 2
        k_phase = nyquist * tf.range(0.0, n_frames) / n_frames

        # Create grid of phases to find corresponding DCT frequencies.
        step_phase_grid = 2 * np.pi / self.n_transients_per_period
        phase_grid = step_phase_grid * tf.range(1.0, n_frequencies + 1) + self.initial_phase_shift
        frequencies = self.tf_interp(phase_grid, phase, k_phase)
        frequencies.set_shape((batch_size, n_frequencies))
        frequencies = frequencies[:,tf.newaxis,:]
        frequencies = tf.repeat(frequencies, repeats=n_frames, axis=1)
        return frequencies
    
    def np_interp(self, phase_grid, phase_ref, k_ref):
        step_phase_grid = 2 * np.pi / self.n_transients_per_period
        fill_value = float(self.sample_rate)
        batch_size = np.shape(phase_ref)[0]
        n_frequencies = np.shape(phase_grid)[0]
        y = np.ndarray((batch_size, n_frequencies), dtype=np.float32)
        for batch in range(batch_size):
            phase_ref_batch = phase_ref[batch,:,0]
            n_shift = np.floor((phase_grid[0] - phase_ref_batch[0]) / step_phase_grid)
            phase_grid_batch = phase_grid - n_shift * step_phase_grid
            y[batch,:] = np.interp(phase_grid_batch, phase_ref_batch, k_ref, left=fill_value, right=fill_value)
        return y

    @tf.function
    def tf_interp(self, phase_grid, phase_ref, k_ref):
        return tf.numpy_function(self.np_interp, [phase_grid, phase_ref, k_ref], tf.float32)

class TransientsDCT(AdditiveDCT):
    '''
    Synthesize audio transients spaced according to the phase.
    '''
    def __init__(self,
                 n_samples=64000,
                 sample_rate=16000,
                 n_transients_per_period=1,
                 initial_phase_shift=0.0,
                 scale_fn=core.exp_sigmoid,
                 name="transients_dct"):
        super().__init__(n_samples, sample_rate, n_transients_per_period,
                         initial_phase_shift, scale_fn, name)
    
    def get_signal(self, *args, **kwargs):
        signal = super().get_signal(*args, **kwargs)
        return tf.signal.idct(signal)

def oscillator_bank(frequency_envelopes,
                    amplitude_envelopes,
                    sample_rate=16000,
                    initial_phases="rand"):
    """Generates audio from sample-wise frequencies for a bank of oscillators.

    Args:
        frequency_envelopes: Sample-wise oscillator frequencies (Hz). Shape
        [batch_size, n_samples, n_sinusoids].
        amplitude_envelopes: Sample-wise oscillator amplitude. Shape [batch_size,
        n_samples, n_sinusoids].
        sample_rate: Sample rate in samples per a second.

    Returns:
        wav: Sample-wise audio. Shape [batch_size, n_samples, n_sinusoids].
    """
    frequency_envelopes = core.tf_float32(frequency_envelopes)
    amplitude_envelopes = core.tf_float32(amplitude_envelopes)
    batch_size = frequency_envelopes.shape[0]
    n_sinusoids = frequency_envelopes.shape[2]

    # Don't exceed Nyquist.
    amplitude_envelopes = core.remove_above_nyquist(frequency_envelopes,
                                                    amplitude_envelopes,
                                                    sample_rate)

    # Change Hz to radians per sample.
    omegas = frequency_envelopes * (2.0 * np.pi)  # rad / sec
    omegas = omegas / float(sample_rate)  # rad / sample

    # Accumulate phase and synthesize.
    phases = core.cumsum(omegas, axis=1)
    if initial_phases == "rand":
        phases += tf.random.uniform((batch_size, 1, n_sinusoids), 
                                    minval=0.0, maxval=2*np.pi)
    wavs = tf.sin(phases)
    harmonic_audio = amplitude_envelopes * wavs  # [mb, n_samples, n_sinusoids]
    audio = tf.reduce_sum(harmonic_audio, axis=-1)  # [mb, n_samples]
    return audio
