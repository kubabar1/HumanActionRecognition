from random import randrange

import numpy as np
from torch.utils.data import Dataset

from ....utils.dataset_util import DatasetInputType, random_rotate_y


class STLSTMDataset(Dataset):
    def __init__(self, data, labels, batch_size, analysed_kpts_description, add_random_rotation_y=False, steps=32, split=20,
                 input_type=DatasetInputType.SPLIT, is_test=False):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.split = split
        self.steps = steps
        self.analysed_kpts_description = analysed_kpts_description
        self.input_type = input_type
        self.add_random_rotation_y = add_random_rotation_y
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_arr = []
        labels_arr = []
        all_analysed_kpts = list(self.analysed_kpts_description.values())
        it = 0

        while it < self.batch_size:
            if self.is_test:
                data_el = self.data[it]
                label_el = self.labels[it]
            else:
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
            elif self.input_type == DatasetInputType.TREE:
                data_arr.append(np.array([kpts_to_tree(a[randrange(len(a))], self.analysed_kpts_description) for a in
                                          np.array_split(data_el[:, :, :], self.split)]))
                labels_arr.append(label_el)
            else:
                raise ValueError('Invalid or unimplemented input type')
            it += 1

        np_data = np.array(data_arr)
        np_label = np.array(labels_arr)

        if self.add_random_rotation_y:
            np_data = random_rotate_y(np_data)

        return np_data, np_label


def kpts_to_tree(kpts, analysed_kpts_description):
    between_hips = (kpts[analysed_kpts_description['right_hip']] + kpts[analysed_kpts_description['left_hip']]) / 2
    between_shoulders = (kpts[analysed_kpts_description['right_shoulder']] + kpts[analysed_kpts_description['left_shoulder']]) / 2
    between_hips_and_shoulders = (between_hips + between_shoulders) / 2

    return np.array([
        between_hips_and_shoulders,
        between_shoulders,
        kpts[analysed_kpts_description['right_shoulder']],
        kpts[analysed_kpts_description['right_elbow']],
        kpts[analysed_kpts_description['right_wrist']],
        kpts[analysed_kpts_description['right_elbow']],
        kpts[analysed_kpts_description['right_shoulder']],
        between_shoulders,
        kpts[analysed_kpts_description['left_shoulder']],
        kpts[analysed_kpts_description['left_elbow']],
        kpts[analysed_kpts_description['left_wrist']],
        kpts[analysed_kpts_description['left_elbow']],
        kpts[analysed_kpts_description['left_shoulder']],
        between_shoulders,
        between_hips_and_shoulders,
        between_hips,
        kpts[analysed_kpts_description['right_hip']],
        kpts[analysed_kpts_description['right_knee']],
        kpts[analysed_kpts_description['right_ankle']],
        kpts[analysed_kpts_description['right_knee']],
        kpts[analysed_kpts_description['right_hip']],
        between_hips,
        kpts[analysed_kpts_description['left_hip']],
        kpts[analysed_kpts_description['left_knee']],
        kpts[analysed_kpts_description['left_ankle']],
        kpts[analysed_kpts_description['left_knee']],
        kpts[analysed_kpts_description['left_hip']],
        between_hips,
        between_hips_and_shoulders
    ])
