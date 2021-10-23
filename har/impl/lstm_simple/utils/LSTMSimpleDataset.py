from random import randrange
from typing import Dict

import numpy as np
from torch.utils.data import Dataset

from .descriptors.jjc import calculate_jjc
from ....utils.dataset_util import DatasetInputType, random_rotate_y, GeometricFeature, SetType, prepare_dataset


class LSTMSimpleDataset(Dataset):
    def __init__(self, data, labels, batch_size, analysed_kpts_description: Dict, set_type: SetType,
                 input_type: DatasetInputType = DatasetInputType.SPLIT,
                 geometric_feature: GeometricFeature = GeometricFeature.JOINT_COORDINATE,
                 steps: int = 32, split: int = 20, add_random_rotation_y: bool = False, is_test: bool = False,
                 use_cache: bool = False, remove_cache: bool = False):
        self.is_test = is_test
        self.geometric_feature = geometric_feature
        self.analysed_kpts_description = analysed_kpts_description
        self.batch_size = batch_size
        self.steps = steps
        self.split = split
        self.input_type = input_type
        self.add_random_rotation_y = add_random_rotation_y
        self.use_cache = use_cache
        self.remove_cache = remove_cache
        self.set_type = set_type

        if geometric_feature == GeometricFeature.JOINT_COORDINATE:
            self.data = data
            self.labels = labels
        else:
            self.data, self.labels = prepare_dataset(data, labels, set_type, self.analysed_kpts_description, geometric_feature,
                                                     use_cache, remove_cache, 'lstm_simple')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_arr = []
        labels_arr = []
        data_len = len(self.data)
        it = 0

        while it < self.batch_size:
            if self.is_test:
                data_el = self.data[it % data_len]
                label_el = self.labels[it % data_len]
                if self.geometric_feature == GeometricFeature.JOINT_COORDINATE:
                    data_el = calculate_jjc(data_el, list(self.analysed_kpts_description.values()))
            else:
                random_data_idx = randrange(self.__len__())
                data_el = self.data[random_data_idx]
                label_el = self.labels[random_data_idx]
                if self.geometric_feature == GeometricFeature.JOINT_COORDINATE:
                    if self.add_random_rotation_y:
                        data_el = random_rotate_y(data_el)
                    data_el = calculate_jjc(data_el, list(self.analysed_kpts_description.values()))

            if self.input_type == DatasetInputType.STEP:
                parts = int(data_el.shape[0] / self.steps)

                for i in range(parts):
                    data_arr.append(data_el[i * self.steps: i * self.steps + self.steps])
                    labels_arr.append(label_el)
                    if it >= self.batch_size:
                        break
                    it += 1
            elif self.input_type == DatasetInputType.SPLIT:
                data_arr.append(np.array([a[randrange(len(a))] for a in np.array_split(data_el[:, :], self.split)]))
                labels_arr.append(label_el)
                it += 1
            else:
                raise ValueError('Invalid or unimplemented input type')

        np_data = np.array(data_arr)
        np_label = np.array(labels_arr)

        return np_data, np_label


def get_input_size(geometric_feature, analysed_kpts_count, is_3d):
    analysed_lines_count = 18
    if geometric_feature == GeometricFeature.JOINT_COORDINATE:
        input_size = analysed_kpts_count * (3 if is_3d else 2)
    elif geometric_feature == GeometricFeature.RELATIVE_POSITION:
        input_size = analysed_kpts_count * (analysed_kpts_count - 1) * 3
    elif geometric_feature == GeometricFeature.JOINT_JOINT_DISTANCE:
        input_size = (analysed_kpts_count * (analysed_kpts_count - 1))
    elif geometric_feature == GeometricFeature.JOINT_JOINT_ORIENTATION:
        input_size = (analysed_kpts_count * (analysed_kpts_count - 1)) * 3
    elif geometric_feature == GeometricFeature.JOINT_LINE_DISTANCE:
        input_size = analysed_lines_count * (analysed_kpts_count - 2)
    elif geometric_feature == GeometricFeature.LINE_LINE_ANGLE:
        input_size = analysed_lines_count * (analysed_lines_count - 1)
    else:
        raise ValueError('Invalid or unimplemented geometric feature type')
    return input_size
