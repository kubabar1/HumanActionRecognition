import math
import os
import shutil
from random import randrange
from typing import Dict

import numpy as np
import tqdm
from torch.utils.data import Dataset

from har.utils.dataset_util import SetType, get_left_kpts, get_right_kpts
from .jtm import rotate, jtm_res_to_pil_img, jtm


class JTMDataset(Dataset):
    def __init__(self, data, labels, image_width, image_height, batch_size, set_type: SetType,
                 analysed_kpts_description: Dict, action_repetitions=100, use_cache=False, remove_cache=False, is_test=False):
        if use_cache:
            self.data, self.labels = generate_jtm_images_dataset(data, labels, image_width, image_height, action_repetitions,
                                                                 set_type, analysed_kpts_description, remove_cache)
        else:
            self.data, self.labels = data, labels
        self.batch_size = batch_size
        self.analysed_kpts_description = analysed_kpts_description
        self.image_width = image_width
        self.image_height = image_height
        self.use_cache = use_cache
        self.remove_cache = remove_cache
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_arr = []
        labels_arr = []

        it = 0
        while it < self.batch_size:
            random_data_idx = randrange(len(self.data))
            if self.is_test:
                data_arr.append(generate_sample_images(self.data[it],
                                                       self.analysed_kpts_description, self.image_width,
                                                       self.image_height))
                labels_arr.append(self.labels[it])
            else:
                if self.use_cache:
                    data_arr.append(np.load(self.data[random_data_idx], allow_pickle=True))
                    labels_arr.append(self.labels[random_data_idx])
                else:
                    smpl_img_front, smpl_img_top, smpl_img_side = generate_sample_images(self.data[random_data_idx],
                                                                                         self.analysed_kpts_description, self.image_width,
                                                                                         self.image_height)
                    data_arr.append([smpl_img_front, smpl_img_top, smpl_img_side])
                    labels_arr.append(self.labels[random_data_idx])
            it += 1

        return data_arr, labels_arr


def generate_jtm_images_dataset(data, labels, image_width, image_height, action_repetitions, set_type, analysed_kpts_description,
                                remove_cache):
    data_base_name = 'data'
    data_root_dir = 'images_cache'
    dataset_cache_dir = os.path.join('dataset_cache', 'jtm', set_type.name)
    data_arr_cache_path = os.path.join(dataset_cache_dir, 'data_arr_cache')
    labels_arr_cache_path = os.path.join(dataset_cache_dir, 'labels_arr_cache')

    if remove_cache and os.path.exists(dataset_cache_dir):
        shutil.rmtree(dataset_cache_dir)

    if os.path.exists(data_arr_cache_path + '.npy') and os.path.exists(labels_arr_cache_path + '.npy'):
        return np.load(data_arr_cache_path + '.npy', allow_pickle=True), np.load(labels_arr_cache_path + '.npy',
                                                                                 allow_pickle=True)

    if not os.path.exists(dataset_cache_dir):
        os.makedirs(dataset_cache_dir)

    if not os.path.exists(os.path.join(dataset_cache_dir, data_root_dir)):
        os.makedirs(os.path.join(dataset_cache_dir, data_root_dir))

    data_arr = []
    labels_arr = []

    actions = {}

    for i in set(labels):
        actions[i] = []

    for idx, lbl in enumerate(labels):
        actions[lbl] += [data[idx]]

    print('Generating dataset ...')
    progress_bar = tqdm.tqdm(total=len(set(labels)) * action_repetitions)
    it = 0
    for a in actions.keys():
        for r in range(action_repetitions):
            ar = actions[a][r % len(actions[a])]

            smpl_img_front, smpl_img_top, smpl_img_side = generate_sample_images(ar, analysed_kpts_description, image_width, image_height)

            np.save(os.path.join(dataset_cache_dir, data_root_dir, data_base_name + '_' + str(it)),
                    [smpl_img_front, smpl_img_top, smpl_img_side])
            data_arr.append(os.path.join(dataset_cache_dir, data_root_dir, data_base_name + '_' + str(it) + '.npy'))
            labels_arr.append(a)
            progress_bar.update(1)
            it += 1

    progress_bar.close()

    np.save(labels_arr_cache_path, labels_arr)
    np.save(data_arr_cache_path, data_arr)

    print('Dataset generated')

    return data_arr, labels_arr


def generate_sample_images(data, analysed_kpts_description, image_width, image_height):
    rotations_degree_x = [0, 15, 30, 45]
    rotations_degree_y = [-45, -30, -15, 0, 15, 30, 45]
    rotations_degree_x_len = len(rotations_degree_x)
    rotations_degree_y_len = len(rotations_degree_y)

    rotation_x = math.radians(rotations_degree_x[randrange(rotations_degree_x_len)])
    rotation_y = math.radians(rotations_degree_y[randrange(rotations_degree_y_len)])

    pos = np.array([np.array([rotate(k, rotation_y, rotation_x) for k in f]) for f in data])

    analysed_kpts_left = get_left_kpts(analysed_kpts_description)
    analysed_kpts_right = get_right_kpts(analysed_kpts_description)
    all_analysed_kpts = list(analysed_kpts_description.values())

    pos_x = (pos[:, all_analysed_kpts, 0] + 1) * image_width / 2
    pos_y = (pos[:, all_analysed_kpts, 1] + 1) * image_height / 2
    pos_z = (pos[:, all_analysed_kpts, 2] + 1) * image_height / 2

    smpl_img_front = jtm_res_to_pil_img(jtm(pos_x, pos_y, image_width, image_height, analysed_kpts_left))
    smpl_img_top = jtm_res_to_pil_img(jtm(pos_x, pos_z, image_width, image_height, analysed_kpts_left))
    smpl_img_side = jtm_res_to_pil_img(jtm(pos_z, pos_y, image_width, image_height, analysed_kpts_left))

    return smpl_img_front, smpl_img_top, smpl_img_side
