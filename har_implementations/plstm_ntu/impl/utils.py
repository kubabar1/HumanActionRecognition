import os
from pathlib import Path
from random import randrange

import numpy as np

classes = [
    "JUMPING_IN_PLACE",
    "JUMPING_JACKS",
    "BENDING_HANDS_UP_ALL_THE_WAY_DOWN",
    "PUNCHING_BOXING",
    "WAVING_TWO_HANDS",
    "WAVING_ONE_HAND_RIGHT",
    "CLAPPING_HANDS",
    "THROWING_A_BALL",
    "SIT_DOWN_THEN_STAND_UP",
    "SIT_DOWN",
    "STAND_UP"
]

video_pose_3d_kpts = {
    'right_wrist': 16,
    'left_wrist': 13,
    'right_elbow': 15,
    'left_elbow': 12,
    'right_shoulder': 14,
    'left_shoulder': 11,
    'right_hip': 1,
    'left_hip': 4,
    'right_knee': 2,
    'left_knee': 5,
    'right_ankle': 3,
    'left_ankle': 6
}


def get_batch(shilouetes_berkeley_path, batch_size=128, training=True, train_threshold=0.8, shilouetes_count=8,
              actions_count=11, repetitions_count=5, split_t=20):
    shilouetes_dirs = sorted(
        [os.path.join(shilouetes_berkeley_path, x.name) for x in Path(shilouetes_berkeley_path).iterdir() if x.is_dir()])
    coordinates_file_name = '3d_coordinates.npy'
    train_size = int(shilouetes_count * train_threshold)
    train_shilouetes = range(train_size)
    test_shilouetes = range(shilouetes_count - train_size)

    data = []
    labels = []
    for i in range(batch_size):
        rand_shilouete_id = randrange(shilouetes_count)
        rand_action_id = randrange(actions_count)
        rand_repetition_id = randrange(repetitions_count)
        coordinates_path = os.path.join(shilouetes_dirs[rand_shilouete_id],
                                        'a' + str(rand_action_id + 1).zfill(2),
                                        'r' + str(rand_repetition_id + 1).zfill(2),
                                        coordinates_file_name)
        data.append(np.array([a[randrange(len(a))] for a in np.array_split(np.load(coordinates_path), split_t)]))

        # if len(pos) < sequence_len:
        #     tmp = []
        #     rp = sequence_len // len(pos)
        #     lt = sequence_len - len(pos) * rp
        #     for _ in range(rp):
        #         tmp.extend(pos)
        #     tmp.extend(pos[:lt])
        #     data.append(np.array(tmp))
        # else:
        #     data.append(pos[:sequence_len])
        labels.append(rand_action_id)

    return np.array(data, dtype='float'), labels


def get_batch2(shilouetes_berkeley_path, sequence_len=100, batch_size=128, training=True, train_threshold=0.8, shilouetes_count=8,
               actions_count=11, repetitions_count=5):
    shilouetes_dirs = sorted(
        [os.path.join(shilouetes_berkeley_path, x.name) for x in Path(shilouetes_berkeley_path).iterdir() if x.is_dir()])
    coordinates_file_name = '3d_coordinates.npy'
    train_size = int(shilouetes_count * train_threshold)
    train_shilouetes = range(train_size)
    test_shilouetes = range(shilouetes_count - train_size)

    data = []
    labels = []
    for i in range(batch_size):
        rand_shilouete_id = randrange(shilouetes_count)
        rand_action_id = randrange(actions_count)
        rand_repetition_id = randrange(repetitions_count)
        coordinates_path = os.path.join(shilouetes_dirs[rand_shilouete_id],
                                        'a' + str(rand_action_id + 1).zfill(2),
                                        'r' + str(rand_repetition_id + 1).zfill(2),
                                        coordinates_file_name)
        pos = np.array(np.load(coordinates_path))

        if len(pos) < sequence_len:
            tmp = []
            rp = sequence_len // len(pos)
            lt = sequence_len - len(pos) * rp
            for _ in range(rp):
                tmp.extend(pos)
            tmp.extend(pos[:lt])
            data.append(np.array(tmp))
        else:
            data.append(pos[:sequence_len])
        labels.append(rand_action_id)

    return np.array(data, dtype='float'), labels
