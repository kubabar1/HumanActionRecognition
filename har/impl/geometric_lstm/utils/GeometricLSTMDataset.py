import os
from random import randrange
from typing import Dict

import numpy as np
import tqdm
from torch.utils.data import Dataset

from .descriptors.jjd import calculate_jjd
from .descriptors.jld import calculate_jld
from .descriptors.rp import calculate_rp
from ....utils.dataset_util import SetType, DatasetInputType, GeometricFeature, random_rotate_y


class GeometricLSTMDataset(Dataset):
    def __init__(self, data, labels, batch_size: int, set_type: SetType, analysed_kpts_description: Dict,
                 geometric_feature: GeometricFeature, input_type: DatasetInputType = DatasetInputType.SPLIT, steps: int = 32,
                 split: int = 20, add_random_rotation_y: bool = False, is_test: bool = False, use_cache: bool = True):
        # self.data, self.labels = generate_geometric_lstm(data, labels, list(analysed_kpts_description.values()), split,
        #                                                  set_type, use_cache)
        self.is_test = is_test
        self.data = data
        self.labels = labels
        self.analysed_kpts_description = analysed_kpts_description
        self.batch_size = batch_size
        self.steps = steps
        self.split = split
        self.input_type = input_type
        self.add_random_rotation_y = add_random_rotation_y
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_arr = []
        labels_arr = []
        all_analysed_kpts = list(self.analysed_kpts_description.values())
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
                    data_arr.append(data_el[i * self.steps: i * self.steps + self.steps, all_analysed_kpts, :])
                    labels_arr.append(label_el)
                    if it >= self.batch_size:
                        break
                    it += 1
            elif self.input_type == DatasetInputType.SPLIT:
                data_arr.append(
                    np.array([a[randrange(len(a))] for a in np.array_split(data_el[:, all_analysed_kpts, :], self.split)]))
                labels_arr.append(label_el)
            else:
                raise ValueError('Invalid or unimplemented input type')
            it += 1

        np_data = np.array(data_arr)
        np_label = np.array(labels_arr)

        if self.add_random_rotation_y:
            np_data = random_rotate_y(np_data)

        return np_data, np_label

    # def __getitem__(self, idx):
    #     data_arr_rp = []
    #     data_arr_jjd = []
    #     data_arr_jld = []
    #     labels_arr = []
    #
    #     while len(labels_arr) < self.batch_size:
    #         random_data_idx = randrange(self.__len__())
    #
    #         data_arr_rp.append(self.data[0][random_data_idx])
    #         data_arr_jjd.append(self.data[1][random_data_idx])
    #         data_arr_jld.append(self.data[2][random_data_idx])
    #         labels_arr.append(self.labels[random_data_idx])
    #
    #     return [np.array(data_arr_rp), np.array(data_arr_jjd), np.array(data_arr_jld)], np.array(labels_arr)


def generate_geometric_lstm(data, labels, all_analysed_kpts, split, set_type, use_cache=True):
    dataset_cache_dir = os.path.join('dataset_cache', 'geometric_multilayer_lstm', set_type.name)
    data_arr_rp_cache_path = os.path.join(dataset_cache_dir, 'data_arr_cache_rp')
    data_arr_jjd_cache_path = os.path.join(dataset_cache_dir, 'data_arr_cache_jjd')
    data_arr_jld_cache_path = os.path.join(dataset_cache_dir, 'data_arr_cache_jld')
    labels_arr_cache_path = os.path.join(dataset_cache_dir, 'labels_arr_cache')

    data_arr_rp, data_arr_jjd, data_arr_jld = [], [], []
    labels_arr = []

    analysed_lines_1 = [[3, 2], [2, 1], [16, 15], [15, 14], [12, 11], [13, 12], [5, 4], [6, 5]]
    analysed_lines_2 = [[3, 1], [16, 14], [13, 11], [6, 4]]
    analysed_lines_3 = [[3, 16], [6, 16], [3, 6], [16, 6], [13, 6], [16, 13]]
    analysed_lines = analysed_lines_1 + analysed_lines_2 + analysed_lines_3

    if use_cache and os.path.exists(data_arr_rp_cache_path + '.npy') and os.path.exists(data_arr_jjd_cache_path + '.npy') \
            and os.path.exists(data_arr_jld_cache_path + '.npy') and os.path.exists(labels_arr_cache_path + '.npy'):
        return [np.load(data_arr_rp_cache_path + '.npy'), np.load(data_arr_jjd_cache_path + '.npy'),
                np.load(data_arr_jld_cache_path + '.npy')], np.load(labels_arr_cache_path + '.npy')

    if not os.path.exists(dataset_cache_dir):
        os.makedirs(dataset_cache_dir)

    print('Generating dataset ...')
    progress_bar = tqdm.tqdm(total=len(data))

    for it in range(len(data)):
        data_el = data[it]
        label_el = labels[it]

        data_el_split = np.array([a[randrange(len(a))] for a in np.array_split(data_el, split)])

        data_rp = calculate_rp(data_el_split, all_analysed_kpts)
        data_jjd = calculate_jjd(data_el_split, all_analysed_kpts)
        data_jld = calculate_jld(data_el_split, all_analysed_kpts, analysed_lines)

        data_arr_rp.append(data_rp)
        data_arr_jjd.append(data_jjd)
        data_arr_jld.append(data_jld)

        labels_arr.append(label_el)
        progress_bar.update(1)

    progress_bar.close()

    np.save(data_arr_rp_cache_path, data_arr_rp)
    np.save(data_arr_jjd_cache_path, data_arr_jjd)
    np.save(data_arr_jld_cache_path, data_arr_jld)
    np.save(labels_arr_cache_path, labels_arr)

    print('Dataset generated')

    return [data_arr_rp, data_arr_jjd, data_arr_jld], labels_arr
