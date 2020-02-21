'''
Utility functionalities.
'''
import time
import matplotlib.pyplot as plt
import numpy as np
import librosa
from librosa.display import specshow

def plot_data_dict(data, save_path=None):
    '''
    Plot the data in a dict.
    '''
    audio_length = len(data["audio"])/data["sample_rate"]
    audio_times = np.arange(0., audio_length, 1/data["sample_rate"])
    input_length = len(data["inputs"]["f0"])/data["frame_rate"]
    input_times = np.arange(0., input_length, 1/data["frame_rate"])
    n_rows = 1 + len(data["inputs"])
    _, ax = plt.subplots(n_rows, 1, figsize=(15, 2*n_rows))
    ax[0].plot(audio_times, data["audio"])
    ax[0].set_title("audio")
    for i, c in enumerate(data["inputs"].keys()):
        ax[i+1].plot(input_times, data["inputs"][c])
        ax[i+1].set_title(c)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)

def plot_audio_f0(audio, audio_rate, f0, f0_rate, fmax=8192, n_fft=8192, n_mels=1024):
    S = librosa.feature.melspectrogram(y=audio, sr=audio_rate, n_fft=n_fft, n_mels=n_mels, fmax=fmax)
    S_dB = librosa.power_to_db(S, ref=np.max)
    ax = specshow(S_dB, x_axis='time', y_axis='mel', sr=audio_rate, fmax=fmax)
    secs = len(audio)/audio_rate
    t_f0 = np.arange(0., secs, 1./f0_rate)
    f0_h, = ax.plot(t_f0, f0, "--")
    ax.set_ylim((0, 5*np.max(f0)))
    ax.set_xlabel("time [s]")
    ax.set_ylabel("frequency [Hz]")
    ax.legend([f0_h], ["f0"], loc="upper right")
    plt.tight_layout()

def get_timestamp():
    return time.strftime("%y%m%d_%H%M%S", time.localtime())

class TimedList:
    '''
    List with added timestamps for each entry.
    '''
    def __init__(self, name=None, debug=True, values=None, times=None):
        self.name = name
        self.debug = debug
        self.values = list() if values is None else values
        self.times = list() if times is None else times
        if (values is not None and times is None) or (values is None and times is not None):
            raise Exception("Both values and times must be given.")
        if values is not None and times is not None:
            if len(values) != len(times):
                raise Exception("values and times must be of equal length.")
    
    def append(self, e):
        now = time.time()
        if self.debug:
            print(self.name, "=\t", e, "\tat time", now)
        self.values.append(e)
        self.times.append(now)