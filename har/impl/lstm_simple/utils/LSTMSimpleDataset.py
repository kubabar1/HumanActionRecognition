import os
from random import randrange
from typing import Dict

import numpy as np
import tqdm
from torch.utils.data import Dataset

from .descriptors.jjc import calculate_jjc
from .descriptors.jjd import calculate_jjd
from .descriptors.jjo import calculate_jjo
from .descriptors.jld import calculate_jld
from .descriptors.lla import calculate_lla
from .descriptors.rp import calculate_rp
from ....utils.dataset_util import DatasetInputType, random_rotate_y, GeometricFeature, SetType


def prepare_dataset(data, labels, set_type, analysed_kpts_description, geometric_feature, use_cache):
    dataset_cache_root_dir = 'dataset_cache'
    dataset_cache_method_dir = 'lstm_simple'
    dataset_cache_data_file_name = 'data_arr_cache'
    dataset_cache_labels_file_name = 'labels_arr_cache'
    dataset_cache_dir = os.path.join(dataset_cache_root_dir, dataset_cache_method_dir, geometric_feature.value, set_type.name)
    data_arr_cache_path = os.path.join(dataset_cache_dir, dataset_cache_data_file_name)
    labels_arr_cache_path = os.path.join(dataset_cache_dir, dataset_cache_labels_file_name)

    if use_cache and os.path.exists(data_arr_cache_path + '.npy') and os.path.exists(labels_arr_cache_path + '.npy'):
        data_arr = np.load(data_arr_cache_path + '.npy', allow_pickle=True)
        labels_arr = np.load(labels_arr_cache_path + '.npy', allow_pickle=True)
    else:
        print('Generating dataset ...')
        progress_bar = tqdm.tqdm(total=len(data))

        data_arr = []
        labels_arr = []

        for it in range(len(data)):
            data_arr.append(get_data_for_geometric_type(data[it], analysed_kpts_description, geometric_feature))
            labels_arr.append(labels[it])
            progress_bar.update(1)
        progress_bar.close()

        if use_cache:
            if not os.path.exists(dataset_cache_dir):
                os.makedirs(dataset_cache_dir)
            np.save(data_arr_cache_path, np.array(data_arr, dtype=object))
            np.save(labels_arr_cache_path, np.array(labels_arr, dtype=object))

        print('Dataset generated')

    return data_arr, labels_arr


def get_data_for_geometric_type(data_el, analysed_kpts_description, geometric_feature):
    all_analysed_kpts = list(analysed_kpts_description.values())
    if geometric_feature == GeometricFeature.RELATIVE_POSITION:
        return calculate_rp(data_el, all_analysed_kpts)
    elif geometric_feature == GeometricFeature.JOINT_JOINT_DISTANCE:
        return calculate_jjd(data_el, all_analysed_kpts)
    elif geometric_feature == GeometricFeature.JOINT_JOINT_ORIENTATION:
        return calculate_jjo(data_el, all_analysed_kpts)
    elif geometric_feature == GeometricFeature.JOINT_LINE_DISTANCE:
        return calculate_jld(data_el, all_analysed_kpts, get_analysed_lines_ids(analysed_kpts_description))
    elif geometric_feature == GeometricFeature.LINE_LINE_ANGLE:
        return calculate_lla(data_el, get_analysed_lines_ids(analysed_kpts_description))
    else:
        raise ValueError('Invalid or unimplemented geometric feature type')


def get_analysed_lines_ids(analysed_kpts_description):
    analysed_lines_1 = [
        [analysed_kpts_description['right_wrist'], analysed_kpts_description['right_elbow']],
        [analysed_kpts_description['right_elbow'], analysed_kpts_description['right_shoulder']],
        [analysed_kpts_description['left_wrist'], analysed_kpts_description['left_elbow']],
        [analysed_kpts_description['left_elbow'], analysed_kpts_description['left_shoulder']],
        [analysed_kpts_description['right_hip'], analysed_kpts_description['right_knee']],
        [analysed_kpts_description['right_knee'], analysed_kpts_description['right_ankle']],
        [analysed_kpts_description['left_hip'], analysed_kpts_description['left_knee']],
        [analysed_kpts_description['left_knee'], analysed_kpts_description['left_ankle']]
    ]
    analysed_lines_2 = [
        [analysed_kpts_description['right_wrist'], analysed_kpts_description['right_shoulder']],
        [analysed_kpts_description['left_wrist'], analysed_kpts_description['left_shoulder']],
        [analysed_kpts_description['right_hip'], analysed_kpts_description['right_ankle']],
        [analysed_kpts_description['left_hip'], analysed_kpts_description['left_ankle']]
    ]
    analysed_lines_3 = [
        [analysed_kpts_description['right_wrist'], analysed_kpts_description['left_wrist']],
        [analysed_kpts_description['right_ankle'], analysed_kpts_description['left_ankle']],
        [analysed_kpts_description['right_wrist'], analysed_kpts_description['right_ankle']],
        [analysed_kpts_description['right_wrist'], analysed_kpts_description['left_ankle']],
        [analysed_kpts_description['left_wrist'], analysed_kpts_description['right_ankle']],
        [analysed_kpts_description['left_wrist'], analysed_kpts_description['left_ankle']]
    ]
    return analysed_lines_1 + analysed_lines_2 + analysed_lines_3


class LSTMSimpleDataset(Dataset):
    def __init__(self, data, labels, batch_size, analysed_kpts_description: Dict, set_type: SetType,
                 input_type: DatasetInputType = DatasetInputType.SPLIT,
                 geometric_feature: GeometricFeature = GeometricFeature.JOINT_COORDINATE,
                 steps: int = 32, split: int = 20, add_random_rotation_y: bool = False, is_test: bool = False,
                 use_cache: bool = False):
        self.is_test = is_test
        self.geometric_feature = geometric_feature
        self.analysed_kpts_description = analysed_kpts_description
        self.batch_size = batch_size
        self.steps = steps
        self.split = split
        self.input_type = input_type
        self.add_random_rotation_y = add_random_rotation_y
        self.use_cache = use_cache
        self.set_type = set_type
        if geometric_feature == GeometricFeature.JOINT_COORDINATE:
            self.data = [calculate_jjc(d, list(analysed_kpts_description.values())) for d in data]
            self.labels = labels
        else:
            self.data, self.labels = prepare_dataset(data, labels, set_type, self.analysed_kpts_description, geometric_feature,
                                                     use_cache)

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
            else:
                random_data_idx = randrange(self.__len__())
                data_el = self.data[random_data_idx]
                label_el = self.labels[random_data_idx]

            if self.input_type == DatasetInputType.STEP:
                parts = int(data_el.shape[0] / self.steps)

                for i in range(parts):
                    data_arr.append(data_el[i * self.steps: i * self.steps + self.steps, :, :])
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

        if self.add_random_rotation_y:
            np_data = random_rotate_y(np_data)

        return np_data, np_label
