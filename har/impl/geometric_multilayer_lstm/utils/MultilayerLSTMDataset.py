import os
from hashlib import sha1
from random import randrange

import numpy as np
import tqdm
from torch.utils.data import Dataset

from har.impl.geometric_multilayer_lstm.utils.descriptors.jjd import calculate_jjd
from har.impl.geometric_multilayer_lstm.utils.descriptors.jld import calculate_jld
from har.impl.geometric_multilayer_lstm.utils.descriptors.rp import calculate_rp
from har.utils.dataset_util import get_analysed_keypoints, SetType


class MultilayerLSTMDataset(Dataset):
    def __init__(self, data, labels, batch_size: int, set_type: SetType, split: int = 20, use_cache: bool = True):
        self.data, self.labels = generate_geometric_multilayer_lstm(data, labels, split, set_type, use_cache)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        data_arr_rp = []
        data_arr_jjd = []
        data_arr_jld = []
        labels_arr = []

        while len(labels_arr) < self.batch_size:
            random_data_idx = randrange(self.__len__())

            data_arr_rp.append(self.data[0][random_data_idx])
            data_arr_jjd.append(self.data[1][random_data_idx])
            data_arr_jld.append(self.data[2][random_data_idx])
            labels_arr.append(self.labels[random_data_idx])

        return [np.array(data_arr_rp), np.array(data_arr_jjd), np.array(data_arr_jld)], np.array(labels_arr)


def generate_geometric_multilayer_lstm(data, labels, split, set_type, use_cache=True):
    dataset_cache_dir = os.path.join('dataset_cache', 'geometric_multilayer_lstm', set_type.name)
    data_arr_rp_cache_path = os.path.join(dataset_cache_dir, 'data_arr_cache_rp')
    data_arr_jjd_cache_path = os.path.join(dataset_cache_dir, 'data_arr_cache_jjd')
    data_arr_jld_cache_path = os.path.join(dataset_cache_dir, 'data_arr_cache_jld')
    labels_arr_cache_path = os.path.join(dataset_cache_dir, 'labels_arr_cache')

    data_arr_rp, data_arr_jjd, data_arr_jld = [], [], []
    labels_arr = []

    analysed_kpts_left, analysed_kpts_right = get_analysed_keypoints()
    all_analysed_kpts = analysed_kpts_left + analysed_kpts_right

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
