import os
import pickle
import numpy as np
from absl import logging
import matplotlib.pyplot as plt
import seaborn as sns

class DataEvaluator:
    SPLITS = ["all", "train", "valid", "test"]

    '''
    Class for evaluating a dataset.
    '''
    def __init__(self, data_provider):
        '''Construct the DataEvaluator.'''
        self.data_provider = data_provider
        self.dataset_name = os.path.basename(data_provider.data_dir)
    
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
        if verbose > 0:
            logging.info("Dataset %s (split: %s) has %d entries (%.1f seconds)." % \
                         (self.dataset_name, self.data_provider.split, n_samples, data["dataset_secs"]))
            logging.info("f0 statistics: max = %.2f; min = %.2f; mean = %.2f; std = %.2f" % \
                         (np.max(data["f0"]), np.min(data["f0"]), np.mean(data["f0"]), np.std(data["f0"])))
        
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
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()
        xlim, ylim = plt.xlim(), plt.ylim()
        plt.close()
        return xlim, ylim

    @staticmethod
    def plot_hist_from_dict(data, input_key="", norm=False, save_path=None,
                            show=True, split="all", xlabel=None, xlim=None,
                            ylim=None, yscale="linear", **fig_kwargs):
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