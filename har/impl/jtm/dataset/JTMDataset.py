import math
import os
from random import randrange

import numpy as np
import tqdm
from torch.utils.data import Dataset

from .jtm import rotate, jtm_res_to_pil_img, jtm


class JTMDataset(Dataset):
    def __init__(self, data, labels, image_width, image_height, analysed_kpts_left, analysed_kpts_right, action_repetitions,
                 batch_size):
        self.data, self.labels = generate_jtm_images_dataset(data, labels, image_width, image_height,
                                                             analysed_kpts_left, analysed_kpts_right, action_repetitions)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_len = len(self.data)

        data_arr = []
        labels_arr = []

        for i in range(self.batch_size):
            random_data_idx = randrange(data_len)
            data_arr.append(np.load(self.data[random_data_idx], allow_pickle=True))
            labels_arr.append(self.labels[random_data_idx])

        return data_arr, labels_arr


def generate_jtm_images_dataset(data, labels, image_width, image_height, analysed_kpts_left, analysed_kpts_right,
                                action_repetitions):
    data_base_name = 'data'
    data_root_dir = 'images_cache'
    dataset_cache_dir = os.path.join('dataset_cache', 'jtm')
    data_arr_cache_path = os.path.join(dataset_cache_dir, 'data_arr_cache')
    labels_arr_cache_path = os.path.join(dataset_cache_dir, 'labels_arr_cache')

    if os.path.exists(data_arr_cache_path + '.npy') and os.path.exists(labels_arr_cache_path + '.npy'):
        # print('Loading dataset from cache ...')
        return np.load(data_arr_cache_path + '.npy', allow_pickle=True), np.load(labels_arr_cache_path + '.npy',
                                                                                 allow_pickle=True)

    if not os.path.exists(dataset_cache_dir):
        os.makedirs(dataset_cache_dir)

    if not os.path.exists(os.path.join(dataset_cache_dir, data_root_dir)):
        os.makedirs(os.path.join(dataset_cache_dir, data_root_dir))

    rotations_degree_x = [0, 15, 30, 45]
    rotations_degree_y = [-45, -30, -15, 0, 15, 30, 45]
    rotations_degree_x_len = len(rotations_degree_x)
    rotations_degree_y_len = len(rotations_degree_y)

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

            rotation_x = math.radians(rotations_degree_x[randrange(rotations_degree_x_len)])
            rotation_y = math.radians(rotations_degree_y[randrange(rotations_degree_y_len)])

            pos = np.array([np.array([rotate(k, rotation_y, rotation_x) for k in f]) for f in ar])

            pos_x = (pos[:, :, 0] + 1) * image_width / 2
            pos_y = (pos[:, :, 1] + 1) * image_height / 2
            pos_z = (pos[:, :, 2] + 1) * image_height / 2

            smpl_img_front = jtm_res_to_pil_img(
                jtm(pos_x, pos_y, image_width, image_height, analysed_kpts_left, analysed_kpts_right))
            smpl_img_top = jtm_res_to_pil_img(
                jtm(pos_x, pos_z, image_width, image_height, analysed_kpts_left, analysed_kpts_right))
            smpl_img_side = jtm_res_to_pil_img(
                jtm(pos_z, pos_y, image_width, image_height, analysed_kpts_left, analysed_kpts_right))

            # show_results_jtm(smpl_img_front, 'results/out_{}'.format(str(rotation_x)), show_results=True, save_img=False)
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
