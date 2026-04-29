import torch
import numpy as np
from torch.utils.data import Dataset
from query_probability import load_ucr
import warnings

# Ignore warnings to keep output clean
warnings.filterwarnings('ignore')


class UcrDataset(Dataset):
    """
    A custom dataset class for handling UCR time series data.

    :param txt_file: Path to the UCR dataset file.
    :param channel_last: Boolean flag to indicate whether the channel dimension should be the last dimension.
    :param normalize: Boolean flag to indicate whether the data should be normalized.
    """

    def __init__(self, txt_file, channel_last, normalize):
        self.data = load_ucr(txt_file, normalize)  # Load and normalize the data
        self.channel_last = channel_last

        # Reshape the data based on the channel_last flag
        if self.channel_last:
            self.data = np.reshape(self.data, [self.data.shape[0], self.data.shape[1], 1])
        else:
            self.data = np.reshape(self.data, [self.data.shape[0], 1, self.data.shape[1]])

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        :param idx: Index of the sample to retrieve.
        :return: A tuple containing the data sample and its corresponding label.
        """
        if not self.channel_last:
            # If channel is not the last dimension, return data and label accordingly
            return self.data[idx, :, 1:], self.data[idx, :, 0]
        else:
            # If channel is the last dimension, return data and label accordingly
            return self.data[idx, 1:, :], self.data[idx, 0, :]

    def get_seq_len(self):
        """
        Returns the sequence length of the time series data (excluding the label).

        :return: Sequence length of the time series data.
        """
        if self.channel_last:
            return self.data.shape[1] - 1
        else:
            return self.data.shape[2] - 1


class AdvDataset(Dataset):
    """
    A custom dataset class for handling adversarial data.

    :param txt_file: Path to the adversarial dataset file.
    """

    def __init__(self, txt_file):
        self.data = np.loadtxt(txt_file)  # Load the adversarial data from a text file
        self.data = np.reshape(self.data, [self.data.shape[0], self.data.shape[1],
                                           1])  # Reshape the data to include a channel dimension

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the adversarial dataset.

        :param idx: Index of the sample to retrieve.
        :return: A tuple containing the data sample and its corresponding label.
        """
        return self.data[idx, 2:, :], self.data[idx, 1, :]

    def get_seq_len(self):
        """
        Returns the sequence length of the adversarial data (excluding the label).

        :return: Sequence length of the adversarial data.
        """
        return self.data.shape[1] - 2


def UCR_dataloader(dataset, batch_size):
    """
    Creates a data loader for the given dataset.

    :param dataset: The dataset from which to load data.
    :param batch_size: Number of samples per batch.
    :return: A DataLoader object for the dataset.
    """
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return data_loader
