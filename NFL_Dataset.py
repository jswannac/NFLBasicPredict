# Class used to build NFL dataset for use in pytorch dataloaders/with the NFL_NN class

import torch
from math import ceil
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np


class NFLDataset(torch.utils.data.Dataset):
    def __init__(self, data_csv_filepath: str):
        self.data = np.loadtxt(data_csv_filepath, delimiter=',')
        self.labels = self.data[:, 0]
        self.scaler = StandardScaler()
        self.game_ids = self.data[:, 1]
        # the 2 below cuts off both point spread/label (which is 0) and game_id. I need to
        # somehow keep game_id and keep it 'with' the relevant data line
        self.data = self.data[:, 2:]
        self.data = self.scaler.fit_transform(self.data)
        self.labels = torch.from_numpy(self.labels)
        self.data = torch.from_numpy(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


def get_train_test_loaders(data_csv_filepath: str, training_percent: float, batch_size: int):
    full_dataset = NFLDataset(data_csv_filepath)
    training_size = ceil(training_percent * len(full_dataset))
    test_size = len(full_dataset) - training_size
    train_data, test_data = random_split(full_dataset, [training_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True)
    return train_loader, test_loader

