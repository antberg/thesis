import os
import pickle
import numpy as np
from absl import logging
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from librosa.display import specshow

class DataEvaluator:
    SPLITS = ["all", "train", "test", "valid"]
    INPUT_UNITS = {"f0": "Hz",
                   "phase": "rad/s",
                   "phase_sub": "rad/s",
                   "phase_sub_sync": "rad/s"}
    LATEX_FROM_KEY = {"f0": "$f_0$",
                      "phase": "$\phi$",
                      "phase_sub": "$\phi$",
                      "phase_sub_sync": "$\phi$"}

    '''
    Class for evaluating a dataset.
    '''
    def __init__(self, data_provider):
        '''Construct the DataEvaluator.'''
        self.data_provider = data_provider
        self.dataset_name = os.path.basename(data_provider.data_dir)
        self.input_stats = dict()
        for input_key in data_provider.input_keys:
            self.input_stats[input_key] = None

    def get_input_stats(self, input_key, stat=None):
        '''Get statistics of input.'''
        if self.input_stats[input_key] is None:
            self.compute_input_stats(input_key)
        return self.input_stats[input_key] if stat is None else self.input_stats[input_key][stat]

    def compute_input_stats(self, input_key):
        '''Compute statistics of input.'''
        dataset = self.data_provider.get_batch(1, repeats=1)
        input_list = []
        for input_tensor in iter(dataset):
            example_numpy = input_tensor[input_key].numpy().flatten()
            input_list.append(example_numpy)
        input_numpy = np.concatenate(input_list)
        self.input_stats[input_key] = dict()
        self.input_stats[input_key]["min"] = np.min(input_numpy)
        self.input_stats[input_key]["max"] = np.max(input_numpy)
        self.input_stats[input_key]["mean"] = np.mean(input_numpy)
        self.input_stats[input_key]["std"] = np.min(input_numpy)
    
    def get_f0_hist_dict(self, bins_f0=None, bins_f0_diff=None, verbose=1):
        '''Get histogram data for f0 and f0_diff.'''
        bins_f0 = 30 if bins_f0 is None else bins_f0
        bins_f0_diff = 30 if bins_f0_diff is None else bins_f0_diff

        # Get data
        data = dict()
        dataset = self.data_provider.get_batch(1, repeats=1)
        input_list = []
        input_diff_list = []
        n_samples = 0
        for input_tensor in iter(dataset):
            example_numpy = input_tensor["f0"].numpy().flatten()
            input_list.append(example_numpy)
            input_diff_list.append(np.diff(example_numpy))
            n_samples += 1
        data["dataset_samples"] = n_samples
        data["dataset_secs"] = n_samples * self.data_provider.example_secs
        data["f0"] = np.concatenate(input_list)
        data["f0_diff"] = np.concatenate(input_diff_list)
        self.input_stats["f0"] = dict(min=np.min(data["f0"]),
                                      max=np.max(data["f0"]),
                                      mean=np.mean(data["f0"]),
                                      std=np.std(data["f0"]))
        if verbose > 0:
            stats = self.input_stats["f0"]
            logging.info("Dataset %s (split: %s) has %d entries (%.1f seconds)." % \
                         (self.dataset_name, self.data_provider.split, n_samples, data["dataset_secs"]))
            logging.info("f0 statistics: min = %.2f; max = %.2f; mean = %.2f; std = %.2f" % \
                         (stats["min"], stats["max"], stats["mean"], stats["std"]))
        
        # Calculate f0 histogram data
        f0_hist_data = self.get_hist_dict_from_numpy(data["f0"], bins_f0, prefix="f0")
        data.update(f0_hist_data)

        # Calculate f0 histogram data normalized by periods
        hist_periods = data["f0_hist_count"] * data["f0_hist_bin_centers"] / self.data_provider.audio_rate
        hist_periods_norm = hist_periods / np.sum(hist_periods)
        data["f0_hist_periods"] = hist_periods
        data["hist_periods_norm"] = hist_periods_norm

        # Calculate f0_diff histogram data
        f0_diff_hist_data = self.get_hist_dict_from_numpy(data["f0_diff"], bins_f0_diff, prefix="f0_diff")
        data.update(f0_diff_hist_data)

        return data
    
    @staticmethod
    def save_data_to_file(data, file_path):
        if os.path.exists(file_path):
            raise ValueError("'%s' already exists!" % file_path)
        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load_data_from_file(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def get_hist_dict_from_numpy(array, bins, prefix=None):
        '''Compute histogram data from numpy array.'''
        prefix = "" if prefix is None else prefix+"_"
        data = dict()

        # Compute histogram data
        hist_count, hist_bins = np.histogram(array, bins=bins, density=False)
        data[prefix + "hist_count"] = hist_count
        data[prefix + "hist_bins"] = hist_bins
        data[prefix + "hist_bin_centers"] = 0.5 * (hist_bins[:-1] + hist_bins[1:])
        data[prefix + "hist_bin_width"] = hist_bins[1] - hist_bins[0]

        # Compute normalized histogram data
        hist_count_norm = hist_count / np.sum(hist_count)
        data[prefix + "hist_count_norm"] = hist_count_norm
        return data
    
    @staticmethod
    def plot_hist(x, y, width, split="all", **fig_kwargs):
        '''Plot histogram.'''
        if fig_kwargs.get("figsize", None) is None:
            fig_kwargs["figsize"] = (4, 3)
        plt.figure(**fig_kwargs)
        sns.set(palette="colorblind")
        color = list(sns.color_palette())[DataEvaluator.SPLITS.index(split)]
        plt.bar(x, y, width=width, color=color)
    
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
    def plot_hist_from_dict(data, input_key="", norm=False, save_path=None,
                            show=True, split="all", xlabel=None, xlim=None,
                            ylim=None, yscale="linear", **fig_kwargs):
        '''Plot input histogram from dict.'''
        prefix = "" if input_key is None else input_key + "_"
        suffix = "_norm" if norm else ""
        y = data["%shist_count%s" % (prefix, suffix)]
        x = data["%shist_bin_centers" % prefix]
        width = data["%shist_bin_width" % prefix]
        DataEvaluator.plot_hist(x, y, width, split=split, **fig_kwargs)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        plt.yscale(yscale)
        plt.xlabel(xlabel)
        return DataEvaluator.plot_postprocess(save_path, show)

    @staticmethod
    def plot_f0_hist_from_file(file_path):
        raise NotImplementedError
    
    def get_example_dict(self, example_id, input_keys="all"):
        '''Get data example dict given an example id.'''
        batch = self.data_provider.get_single_batch(batch_number=example_id)
        data = dict()
        data["audio"] = batch["audio"].numpy().flatten()
        data["audio_rate"] = self.data_provider.audio_rate
        data["input_rate"] = self.data_provider.input_rate
        data["inputs"] = dict()
        data["input_stats"] = dict()
        for input_key in self.data_provider.input_keys:
            if input_keys == "all" or input_key in input_keys:
                data["inputs"][input_key] = batch[input_key].numpy().flatten()
                data["input_stats"][input_key] = self.get_input_stats(input_key)
        return data
    
    @staticmethod
    def plot_inputs_from_dict(data, split="all", save_path=None, show=True, **fig_kwargs):
        '''Plot inputs for a given example.'''
        if fig_kwargs.get("figsize", None) is None:
            fig_kwargs["figsize"] = (4, 4)
        sns.set(palette="colorblind")
        color = list(sns.color_palette())[DataEvaluator.SPLITS.index(split)]
        n_inputs = len(data["inputs"])
        n_subplots = 1 + n_inputs
        _, axes = plt.subplots(n_subplots, 1, **fig_kwargs)
        axes_iter = iter(axes)
        ax = next(axes_iter)
        t = np.arange(0.0, len(data["audio"]))/data["audio_rate"]
        ax.plot(t, data["audio"], color=color)
        ax.set_ylim((-1, 1))
        ax.set_ylabel("audio")
        DataEvaluator.remove_xticks(ax)
        for i, keyval in enumerate(data["inputs"].items()):
            key, values = keyval
            ax = next(axes_iter)
            t = np.arange(0.0, len(values))/data["input_rate"]
            ax.plot(t, values, color=color)
            ylim_min = np.floor(data["input_stats"][key]["min"])
            ylim_max = np.ceil(data["input_stats"][key]["max"])
            ax.set_ylim((ylim_min, ylim_max))
            unit = DataEvaluator.INPUT_UNITS.get(key)
            unit = " [%s]" % unit if unit else ""
            tex_label = DataEvaluator.LATEX_FROM_KEY.get(key)
            tex_label = tex_label if tex_label else key
            ylabel = tex_label + unit
            ax.set_ylabel(ylabel)
            if i + 1 < n_inputs:
                DataEvaluator.remove_xticks(ax)
        ax.set_xlabel("time [s]")
        return DataEvaluator.plot_postprocess(save_path, show)
    
    @staticmethod
    def plot_spectrogram_from_dict(data, spec_type="mel", split="all", plot_f0=False, save_path=None, show=True, **fig_kwargs):
        '''Plot spectrogram for a given example.'''
        if fig_kwargs.get("figsize", None) is None:
            fig_kwargs["figsize"] = (4, 6)
        if spec_type == "mel":
            S_dB = DataEvaluator.get_mel_spectrogram_from_dict(data)
            specshow_kw = dict(y_axis="mel")
        elif spec_type == "cqt":
            fmin = 20.0
            bins_per_octave = 16
            n_octaves = int(np.log(data["audio_rate"] / 2 / fmin) / np.log(2))
            n_bins = bins_per_octave * n_octaves
            cqt_kw = dict(fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
            S_dB = DataEvaluator.get_cqt_spectrogram_from_dict(data, **cqt_kw)
            specshow_kw = cqt_kw
            specshow_kw.update(dict(y_axis="cqt_hz"))
            specshow_kw.pop("n_bins")
        else:
            raise ValueError("%s is not a valid spectrogram type.")
        sns.set(palette="colorblind")
        color = list(sns.color_palette())[DataEvaluator.SPLITS.index(split)]
        _, ax = plt.subplots(1, 1, **fig_kwargs)
        specshow(S_dB, x_axis="time", sr=data["audio_rate"], fmax=data["audio_rate"]/2, ax=ax, **specshow_kw)
        if plot_f0:
            f0 = data["inputs"]["f0"]
            t = np.arange(0.0, len(f0))/data["input_rate"]
            ax.plot(t, f0, "--", color=color, label="$f_0$")
            plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower right")
        ax.set_ylabel("frequency [Hz]")
        ax.set_xlabel("time [s]")
        return DataEvaluator.plot_postprocess(save_path, show)

    @staticmethod
    def get_mel_spectrogram_from_dict(data, n_fft=4092, n_mels=512):
        '''Get Mel spectrogram from dict.'''
        audio = np.asfortranarray(data["audio"])
        S = librosa.feature.melspectrogram(y=audio,
                                           sr=data["audio_rate"],
                                           n_fft=n_fft,
                                           n_mels=n_mels,
                                           fmax=data["audio_rate"]/2)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB

    @staticmethod
    def get_cqt_spectrogram_from_dict(data, **kwargs):
        '''Get constant-Q transform spectrogram from dict.'''
        audio = np.asfortranarray(data["audio"])
        S = np.abs(librosa.cqt(audio, sr=data["audio_rate"], **kwargs))
        S_dB = librosa.amplitude_to_db(S, ref=np.max)
        return S_dB
