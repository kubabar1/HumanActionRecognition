import os
import shutil
from pathlib import Path
from random import randrange

import numpy as np
from .descriptors.jjd import create_jjd_batch
from .descriptors.jld import create_jld_batch
from .descriptors.rp import create_rp_batch
import tqdm

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
    "STAND_UP",
]


def prepare_data_berkeley_mhad(dataset_path, classes, analysed_lines, analysed_kpts, batch_cache_path='batch_cache'):
    shilouetes_dirs = sorted([os.path.join(dataset_path, x.name) for x in Path(dataset_path).iterdir() if x.is_dir()])
    shilouetes_count = len(shilouetes_dirs)
    actions_count = len(classes)
    coordinates_file_name = '3d_coordinates.npy'
    repetitions_count = 5
    progress_bar = tqdm.tqdm(total=shilouetes_count * actions_count * repetitions_count)

    if os.path.exists(batch_cache_path):
        shutil.rmtree(batch_cache_path)

    os.mkdir(batch_cache_path)

    for s in range(shilouetes_count):
        os.mkdir(os.path.join(batch_cache_path, str(s)))
        for a in range(actions_count):
            os.mkdir(os.path.join(batch_cache_path, str(s), str(a)))
            for r in range(repetitions_count):
                os.mkdir(os.path.join(batch_cache_path, str(s), str(a), str(r)))
                progress_bar.update(1)
                coordinates_path = os.path.join(shilouetes_dirs[s],
                                                'a' + str(a + 1).zfill(2),
                                                'r' + str(r + 1).zfill(2),
                                                coordinates_file_name)
                pos = np.load(coordinates_path)
                rp = create_rp_batch(pos, analysed_kpts)
                jjd = create_jjd_batch(pos, analysed_kpts)
                jld = create_jld_batch(pos, analysed_lines, analysed_kpts)

                np.save(os.path.join(batch_cache_path, str(s), str(a), str(r), 'rp'), rp)
                np.save(os.path.join(batch_cache_path, str(s), str(a), str(r), 'jjd'), jjd)
                np.save(os.path.join(batch_cache_path, str(s), str(a), str(r), 'jld'), jld)


def get_batch(batch_cache_path='batch_cache', batch_size=128, split_t=20, training=True):
    rp_batch = []
    jjd_batch = []
    jld_batch = []
    labels = []
    repeats_paths = [(int(action_dir.name), repeat_path) for shilouete_dir in sorted(Path(batch_cache_path).iterdir()) for
                     action_dir in sorted(Path(shilouete_dir).iterdir()) for repeat_path in sorted(Path(action_dir).iterdir())
                     if (int(repeat_path.name) < 4 if training else int(repeat_path.name) == 4)]
    repeats_count = len(repeats_paths)

    for _ in range(batch_size):
        action_id, random_repeat = repeats_paths[randrange(repeats_count)]

        rp_path = os.path.join(random_repeat, 'rp.npy')
        jjd_path = os.path.join(random_repeat, 'jjd.npy')
        jld_path = os.path.join(random_repeat, 'jld.npy')

        rp = np.array([a[randrange(len(a))] for a in np.array_split(np.load(rp_path), split_t)])
        jjd = np.array([a[randrange(len(a))] for a in np.array_split(np.load(jjd_path), split_t)])
        jld = np.array([a[randrange(len(a))] for a in np.array_split(np.load(jld_path), split_t)])

        rp_batch.append(rp)
        jjd_batch.append(jjd)
        jld_batch.append(jld)
        labels.append(action_id)
    return np.array(rp_batch), np.array(jjd_batch), np.array(jld_batch), np.array(labels)


def get_test_data(batch_cache_path='batch_cache', split_t=20):
    rp_batch = []
    jjd_batch = []
    jld_batch = []
    labels = []

    repeats_paths = [(int(action_dir.name), repeat_path) for shilouete_dir in sorted(Path(batch_cache_path).iterdir()) for
                     action_dir in sorted(Path(shilouete_dir).iterdir()) for repeat_path in sorted(Path(action_dir).iterdir())
                     if int(repeat_path.name) == 4]

    for action_id, r_path in repeats_paths:
        rp_path = os.path.join(r_path, 'rp.npy')
        jjd_path = os.path.join(r_path, 'jjd.npy')
        jld_path = os.path.join(r_path, 'jld.npy')

        rp = np.array([a[randrange(len(a))] for a in np.array_split(np.load(rp_path), split_t)])
        jjd = np.array([a[randrange(len(a))] for a in np.array_split(np.load(jjd_path), split_t)])
        jld = np.array([a[randrange(len(a))] for a in np.array_split(np.load(jld_path), split_t)])

        rp_batch.append(rp)
        jjd_batch.append(jjd)
        jld_batch.append(jld)

        labels.append(action_id)
    return np.array(rp_batch), np.array(jjd_batch), np.array(jld_batch), np.array(labels)
