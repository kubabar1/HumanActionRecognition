import os
from pathlib import Path
from random import randrange
import numpy as np
from scipy.spatial import distance
from impl.utils import to_hue, bilinear_resize, to_PIL_img


def jdm(positions):
    # analysed_kpts_left = [4, 5, 6, 11, 12, 13]
    # analysed_kpts_right = [1, 2, 3, 14, 15, 16]
    analysed_kpts_all = [16, 15, 14, 11, 12, 13, 4, 5, 6, 1, 2, 3]

    channel1 = []
    channel2 = []
    channel3 = []
    channel4 = []

    for frame in positions:
        c1 = []
        c2 = []
        c3 = []
        c4 = []

        an_kpts = np.concatenate((np.array(frame[analysed_kpts_all]), np.array(frame[analysed_kpts_all])))

        for i in range(an_kpts.shape[0] - 1):
            for j in range(i + 1, an_kpts.shape[0]):
                kpt1 = an_kpts[i]
                kpt2 = an_kpts[j]
                c1.append(distance.euclidean((kpt1[0], kpt1[1]), (kpt2[0], kpt2[1])))
                c2.append(distance.euclidean((kpt1[0], kpt1[2]), (kpt2[0], kpt2[2])))
                c3.append(distance.euclidean((kpt1[1], kpt1[2]), (kpt2[1], kpt2[2])))
                c4.append(distance.euclidean(kpt1, kpt2))

        channel1.append(np.array(c1))
        channel2.append(np.array(c2))
        channel3.append(np.array(c3))
        channel4.append(np.array(c4))

    fc = len(channel1)

    channel1 = np.array(channel1).reshape((-1, fc))
    channel2 = np.array(channel2).reshape((-1, fc))
    channel3 = np.array(channel3).reshape((-1, fc))
    channel4 = np.array(channel4).reshape((-1, fc))

    return channel1, channel2, channel3, channel4


def get_mini_batch(shilouetes_berkeley_path, classes, samples_count=256, training=True):
    id = 0
    # positions = np.array([np.array([rotate(k, math.radians(rotate_y), math.radians(rotate_x)) for k in f]) for f in np.load(p_path)])
    shilouetes_dirs = sorted(
        [os.path.join(shilouetes_berkeley_path, x.name) for x in Path(shilouetes_berkeley_path).iterdir() if x.is_dir()])
    shilouetes_count = len(shilouetes_dirs)
    actions_count = len(classes)
    coordinates_file_name = '3d_coordinates.npy'

    data = []
    labels = []

    h_min = 0
    h_max = 1

    for i in range(samples_count):
        print(id)
        id += 1
        rand_shilouete_id = randrange(shilouetes_count)
        rand_action_id = randrange(actions_count)
        rand_repetition_id = randrange(4) if training else 4
        coordinates_path = os.path.join(shilouetes_dirs[rand_shilouete_id],
                                        'a' + str(rand_action_id + 1).zfill(2),
                                        'r' + str(rand_repetition_id + 1).zfill(2),
                                        coordinates_file_name)
        pos = np.load(coordinates_path)
        channel1, channel2, channel3, channel4 = jdm(pos)
        res_width = 200
        res_height = channel1.shape[0]

        channel1 = to_hue(bilinear_resize(np.array(channel1), res_height, res_width), h_min, h_max)
        channel2 = to_hue(bilinear_resize(np.array(channel2), res_height, res_width), h_min, h_max)
        channel3 = to_hue(bilinear_resize(np.array(channel3), res_height, res_width), h_min, h_max)
        channel4 = to_hue(bilinear_resize(np.array(channel4), res_height, res_width), h_min, h_max)

        sample_img_xy = to_PIL_img(channel1)
        sample_img_xz = to_PIL_img(channel2)
        sample_img_yz = to_PIL_img(channel3)
        sample_img_xyz = to_PIL_img(channel4)
        data.append([sample_img_xy, sample_img_xz, sample_img_yz, sample_img_xyz])
        labels.append(rand_action_id)
    return data, labels
