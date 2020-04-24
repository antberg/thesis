'''
Utility funcions.
'''
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import librosa
from librosa.display import specshow
import seaborn as sns

class Util:
    '''
    Utility class with methods for storing files, postprocessing plots etc.
    '''
    PLOT_PALETTE = "colorblind"
    SPLITS = ["all", "train", "test", "valid"]

    # =====================
    # FILE STREAM UTILITIES
    # =====================
    @staticmethod
    def save_data_to_file(data, file_path):
        '''Pickle data.'''
        if os.path.exists(file_path):
            raise ValueError("'%s' already exists!" % file_path)
        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load_data_from_file(file_path):
        '''Load pickled data.'''
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data
    
    # ==============
    # PLOT UTILITIES
    # ==============
    @staticmethod
    def plot_postprocess(save_path=None, show=True):
        '''Postprocess current plot.'''
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()
        xlim, ylim = plt.xlim(), plt.ylim()
        plt.close()
        return xlim, ylim
    
    @staticmethod
    def remove_xticks(ax=plt):
        ax.tick_params(axis="x",          # changes apply to the x-axis
                       which="both",      # both major and minor ticks are affected
                       bottom=False,      # ticks along the bottom edge are off
                       top=False,         # ticks along the top edge are off
                       labelbottom=False) # labels along the bottom edge are off
    
    @staticmethod
    def plot_spectrogram_from_dict(data, audio_key="audio",
                                         spec_type="mel",
                                         split="all",
                                         plot_f0=False,
                                         save_path=None,
                                         show=True, **fig_kwargs):
        '''Plot spectrogram for a given example.'''
        if fig_kwargs.get("figsize", None) is None:
            fig_kwargs["figsize"] = (4, 6)
        if spec_type == "mel":
            S_dB = Util.get_mel_spectrogram(data[audio_key], data["audio_rate"])
            specshow_kw = dict(y_axis="mel")
        elif spec_type == "cqt":
            fmin = 20.0
            bins_per_octave = 16
            n_octaves = int(np.log(data["audio_rate"] / 2 / fmin) / np.log(2))
            n_bins = bins_per_octave * n_octaves
            cqt_kw = dict(fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
            S_dB = Util.get_cqt_spectrogram(data[audio_key], data["audio_rate"], **cqt_kw)
            specshow_kw = cqt_kw
            specshow_kw.update(dict(y_axis="cqt_hz"))
            specshow_kw.pop("n_bins")
        else:
            raise ValueError("%s is not a valid spectrogram type.")
        sns.set(palette="colorblind")
        color = list(sns.color_palette())[Util.SPLITS.index(split)]
        _, ax = plt.subplots(1, 1, **fig_kwargs)
        specshow(S_dB, x_axis="time", sr=data["audio_rate"], fmax=data["audio_rate"]/2, ax=ax, **specshow_kw)
        if plot_f0:
            f0 = data["inputs"]["f0"]
            t = np.arange(0.0, len(f0))/data["input_rate"]
            ax.plot(t, f0, "--", color=color, label="$f_0$")
            plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower right")
        ax.set_ylabel("frequency [Hz]")
        ax.set_xlabel("time [s]")
        return Util.plot_postprocess(save_path, show)

    # =====================
    # SPECTROGRAM UTILITIES
    # =====================

    @staticmethod
    def get_mel_spectrogram(audio, sample_rate=48000, n_fft=4092, n_mels=512):
        '''Get Mel spectrogram from dict.'''
        audio = np.asfortranarray(audio)
        S = librosa.feature.melspectrogram(y=audio,
                                           sr=sample_rate,
                                           n_fft=n_fft,
                                           n_mels=n_mels,
                                           fmax=sample_rate/2)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB

    @staticmethod
    def get_cqt_spectrogram(audio, sample_rate=48000, scale="db", **kwargs):
        '''Get constant-Q transform spectrogram from dict.'''
        audio = np.asfortranarray(audio)
        S = np.abs(librosa.cqt(audio, sr=sample_rate, **kwargs))
        if scale == "amp":
            return S
        if scale == "db":
            S_dB = librosa.amplitude_to_db(S, ref=np.max)
            return S_dB
        raise ValueError("%s is not a valid scale." % scale)
