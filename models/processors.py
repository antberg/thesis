'''
DDSP processor groups representing different signal models.
'''
from ddsp.synths import Additive, FilteredNoise
from ddsp.processors import ProcessorGroup, Add

from .synths import Transients, TransientsDCT

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

    '''#TEMP CODE FOR INSPECTING ADDITIVE AND SUBTRACTIVE COMPONENTS
    def call(self, dag_inputs):
        dag_outputs = self.get_controls(dag_inputs)
        
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.plot(dag_inputs["additive"]["controls"]["amplitudes"].numpy()[0,:,0])
        plt.show()
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.imshow(dag_inputs["additive"]["controls"]["harmonic_distribution"].numpy()[0,:,:].T, origin="lower")
        ax.set_aspect("auto")
        plt.show()
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.imshow(dag_inputs["subtractive"]["controls"]["magnitudes"].numpy()[0,:,:].T, origin="lower")
        ax.set_aspect("auto")
        plt.show()
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.plot(dag_inputs["f0"].numpy()[0,:])
        plt.show()
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.plot(dag_inputs["f0_scaled"].numpy()[0,:])
        plt.show()
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.plot(dag_inputs["f0_sub_scaled"].numpy()[0,:])
        plt.show()
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.plot(dag_inputs["osc_scaled"].numpy()[0,:])
        plt.show()

        additive_signal = dag_outputs["additive"]["signal"]
        subtractive_signal = dag_outputs["subtractive"]["signal"]
        signal = self.get_signal(dag_outputs)
        return signal'''
    
class HarmonicPlusNoisePlusTransients(ProcessorGroup):
    '''Spectral modeling synthesis (SMS) with harmonic sinusoids.'''
    def __init__(self, window_secs=None, audio_rate=None, input_rate=None, name="harmonic_plus_noise_plus_transients"):
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
        self.transients = Transients(n_samples=self.n_samples, sample_rate=self.audio_rate, name="transients")
        self.hpn = Add(name="hpn")
        self.hpnt = Add(name="hpnt")
        dag = [(self.additive, ["amps", "harmonic_distribution", "f0"]),
               (self.subtractive, ["noise_magnitudes"]),
               (self.transients, ["transient_amps", "transient_distribution"]),
               (self.hpn, ["additive/signal", "subtractive/signal"]),
               (self.hpnt, ["hpn/signal", "transients/signal"])]
        super(HarmonicPlusNoisePlusTransients, self).__init__(dag, name)

    '''#TEMP CODE FOR INSPECTING ADDITIVE AND SUBTRACTIVE COMPONENTS
    def call(self, dag_inputs):
        dag_outputs = self.get_controls(dag_inputs)
        
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.plot(dag_inputs["additive"]["controls"]["amplitudes"].numpy()[0,:,0])
        plt.show()
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.imshow(dag_inputs["additive"]["controls"]["harmonic_distribution"].numpy()[0,:,:].T, origin="lower")
        ax.set_aspect("auto")
        plt.show()
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.imshow(dag_inputs["subtractive"]["controls"]["magnitudes"].numpy()[0,:,:].T, origin="lower")
        ax.set_aspect("auto")
        plt.show()
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.plot(dag_inputs["transients"]["controls"]["amplitudes"].numpy()[0,:,0])
        plt.show()
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.imshow(dag_inputs["transients"]["controls"]["frequency_distribution"].numpy()[0,:,:].T, origin="lower")
        ax.set_aspect("auto")
        plt.show()
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.plot(dag_inputs["f0"].numpy()[0,:])
        plt.show()
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.plot(dag_inputs["f0_scaled"].numpy()[0,:])
        plt.show()
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.plot(dag_inputs["f0_sub_scaled"].numpy()[0,:])
        plt.show()
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.plot(dag_inputs["osc_scaled"].numpy()[0,:])
        plt.show()

        additive_signal = dag_outputs["additive"]["signal"]
        subtractive_signal = dag_outputs["subtractive"]["signal"]
        transient_signal = dag_outputs["transients"]["signal"]
        signal = self.get_signal(dag_outputs)
        return signal'''

class HarmonicPlusTransients(ProcessorGroup):
    '''Spectral modeling synthesis (SMS) with harmonic sinusoids.'''
    def __init__(self, window_secs=None, audio_rate=None, input_rate=None, name="harmonic_plus_transients"):
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
        self.transients = Transients(n_samples=self.n_samples, sample_rate=self.audio_rate, name="transients")
        self.hpt = Add(name="hpt")
        dag = [(self.additive, ["amps", "harmonic_distribution", "f0"]),
               (self.transients, ["transient_amps", "transient_distribution"]),
               (self.hpt, ["additive/signal", "transients/signal"])]
        super(HarmonicPlusTransients, self).__init__(dag, name)

    '''#TEMP CODE FOR INSPECTING ADDITIVE AND SUBTRACTIVE COMPONENTS
    def call(self, dag_inputs):
        dag_outputs = self.get_controls(dag_inputs)

        additive_signal = dag_outputs["additive"]["signal"]
        transient_signal = dag_outputs["transients"]["signal"]
        signal = self.get_signal(dag_outputs)
        return additive_signal'''

class HarmonicPlusTransientsPhase(ProcessorGroup):
    '''
    Harmonic-plus-transients model with transients placed on times
    correseponding to constant phases.
    '''
    def __init__(self, window_secs=None,
                       audio_rate=None,
                       input_rate=None,
                       n_transients_per_period=1,
                       initial_phase_shift=0.0,
                       name="harmonic_plus_transients_phase"):
        if window_secs is None:
            raise ValueError("Length of windows (window_secs) must be set.")
        if audio_rate is None:
            raise ValueError("Audio sample rate (audio_rate) must be set.")
        if input_rate is None:
            raise ValueError("Input sample rate (input_rate) must be set.")
        self.window_secs = window_secs
        self.audio_rate = audio_rate
        self.input_rate = input_rate
        self.n_transients_per_period = n_transients_per_period
        self.initial_phase_shift = initial_phase_shift
        self.n_samples = int(window_secs * audio_rate)
        self.additive = Additive(n_samples=self.n_samples,
                                 sample_rate=self.audio_rate,
                                 name="additive")
        self.transients = TransientsDCT(n_samples=self.n_samples,
                                        sample_rate=self.audio_rate,
                                        n_transients_per_period=self.n_transients_per_period,
                                        initial_phase_shift=self.initial_phase_shift,
                                        name="transients")
        self.add = Add(name="hptp")
        dag = [(self.additive, ["amps", "harmonic_distribution", "f0"]),
               (self.transients, ["transient_amps", "transient_distribution", "phase_unwrapped"]),
               (self.add, ["additive/signal", "transients/signal"])]
        super(HarmonicPlusTransientsPhase, self).__init__(dag, name)

    '''#TEMP CODE FOR INSPECTING INDIVIDUAL SIGNAL COMPONENTS
    def call(self, dag_inputs):
        dag_outputs = self.get_controls(dag_inputs)
        
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.plot(dag_inputs["phase"].numpy()[0,:])
        plt.show()
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.plot(dag_inputs["phase_unwrapped"].numpy()[0,:])
        plt.show()
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.plot(dag_inputs["transients"]["controls"]["amplitudes"].numpy()[0,:,0])
        plt.show()
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.imshow(dag_inputs["transients"]["controls"]["frequency_distribution"].numpy()[0,:,:].T, origin="lower")
        ax.set_aspect("auto")
        plt.show()

        additive_signal = dag_outputs["additive"]["signal"]
        transient_signal = dag_outputs["transients"]["signal"]
        signal = self.get_signal(dag_outputs)
        return additive_signal'''
