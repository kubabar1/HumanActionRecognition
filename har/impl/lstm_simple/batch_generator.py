import os
from random import randrange

import numpy as np


def get_batch_ntu_rgbd(dataset_path, batch_size=128, steps=16, is_training=True, data_npy_file_name='3d_coordinates.npy'):
    scenes_count = 1
    cameras_count = 3
    persons_count = 8
    repetitions_count = 2

    excluded_actions = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
    all_actions = list(range(1, 61))

    for a in excluded_actions:
        all_actions.remove(a)

    scenes_id = scenes_count

    data_arr = []
    labels_arr = []

    analysed_kpts_left = [4, 5, 6, 11, 12, 13]
    analysed_kpts_right = [1, 2, 3, 14, 15, 16]
    all_analysed_kpts = analysed_kpts_left + analysed_kpts_right

    while len(data_arr) < batch_size:
        camera_id = randrange(cameras_count) + 1
        person_id = randrange(6) + 1 if is_training else randrange(6, persons_count) + 1
        repetition_id = randrange(repetitions_count) + 1
        action_id = randrange(len(all_actions))

        selected_dir = 'S{}C{}P{}R{}A{}_rgb'.format(str(scenes_id).zfill(3), str(camera_id).zfill(3), str(person_id).zfill(3),
                                                    str(repetition_id).zfill(3), str(all_actions[action_id]).zfill(3))
        data_path = os.path.join(dataset_path, selected_dir, data_npy_file_name)
        data = np.array(np.load(data_path))
        label = all_actions[action_id]
        parts = int(data.shape[0] / steps)

        for i in range(parts):
            data_arr.append(data[i * steps: i * steps + steps, all_analysed_kpts, :])
            labels_arr.append(label)
            if len(data_arr) >= batch_size:
                break
    return np.array(data_arr), np.array(labels_arr)
