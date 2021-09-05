from random import randrange

import numpy as np
from torch.utils.data import Dataset

from ....utils.dataset_util import DatasetInputType, random_rotate_y


class LSTMSimpleDataset(Dataset):
    def __init__(self, data, labels, batch_size, analysed_kpts_description, input_type=DatasetInputType.STEP, steps=32, split=20,
                 add_random_rotation_y=False):
        self.data = data
        self.labels = labels
        self.analysed_kpts_description = analysed_kpts_description
        self.batch_size = batch_size
        self.steps = steps
        self.split = split
        self.input_type = input_type
        self.add_random_rotation_y = add_random_rotation_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_arr = []
        labels_arr = []
        all_analysed_kpts = list(self.analysed_kpts_description.values())

        while len(data_arr) < self.batch_size:
            random_data_idx = randrange(self.__len__())
            data_el = self.data[random_data_idx]
            label_el = self.labels[random_data_idx]

            if self.input_type == DatasetInputType.STEP:
                parts = int(data_el.shape[0] / self.steps)

                for i in range(parts):
                    data_arr.append(data_el[i * self.steps: i * self.steps + self.steps, all_analysed_kpts, :])
                    labels_arr.append(label_el)
                    if len(data_arr) >= self.batch_size:
                        break
            elif self.input_type == DatasetInputType.SPLIT:
                data_arr.append(
                    np.array([a[randrange(len(a))] for a in np.array_split(data_el[:, all_analysed_kpts, :], self.split)]))
                labels_arr.append(label_el)
            else:
                raise ValueError('Invalid or unimplemented input type')

        np_data = np.array(data_arr)
        np_label = np.array(labels_arr)

        if self.add_random_rotation_y:
            np_data = random_rotate_y(np_data)

        return np_data, np_label
