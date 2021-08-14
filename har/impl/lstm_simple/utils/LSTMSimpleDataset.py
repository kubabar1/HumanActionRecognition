from random import randrange

import numpy as np
from torch.utils.data import Dataset


class LSTMSimpleDataset(Dataset):
    def __init__(self, data, labels, batch_size, steps=32):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.steps = steps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_arr = []
        labels_arr = []

        while len(data_arr) < self.batch_size:
            random_data_idx = randrange(self.__len__())
            data_el = self.data[random_data_idx]
            label_el = self.labels[random_data_idx]
            parts = int(data_el.shape[0] / self.steps)

            for i in range(parts):
                data_arr.append(data_el[i * self.steps: i * self.steps + self.steps, :, :])
                labels_arr.append(label_el)
                if len(data_arr) >= self.batch_size:
                    break
        return np.array(data_arr), np.array(labels_arr)
