from pathlib import Path
from random import randrange
import numpy as np
from scipy.spatial import distance
import colorsys
import os
import time
import math
from utils import draw_circle, show_results_jtm, rotate, jtm_res_to_PIL_img


def jtm(positions_w, positions_h, image_width, image_height, L=1, s_min=0, s_max=1, b_min=0, b_max=1):
    img = np.zeros((image_height, image_width, 3))
    frames_count = len(positions_w)
    kpts_count = len(positions_w[0])
    analysed_kpts_left = [4, 5, 6, 11, 12, 13]
    analysed_kpts_right = [1, 2, 3, 14, 15, 16]
    all_analysed_kpts = analysed_kpts_left + analysed_kpts_right
    hue = np.zeros([frames_count])
    v = np.zeros([frames_count, kpts_count])
    pos_tmp = np.stack([positions_w, positions_h], axis=2)

    for frame_id, kpts in enumerate(pos_tmp):
        hue[frame_id] = (frame_id / (frames_count - 1)) * L
        for kpt_id, _ in enumerate(kpts):
            if frame_id < frames_count - 1:
                start = (positions_h[frame_id + 1][kpt_id], positions_w[frame_id + 1][kpt_id])
                end = (positions_h[frame_id][kpt_id], positions_w[frame_id][kpt_id])
            else:
                start = (positions_w[frame_id][kpt_id], positions_h[frame_id][kpt_id])
                end = (positions_w[frame_id - 1][kpt_id], positions_h[frame_id - 1][kpt_id])
            v[frame_id][kpt_id] = distance.euclidean(start, end)

    saturation = v / np.max(v) * (s_max - s_min) + s_min
    brightness = v / np.max(v) * (b_max - b_min) + b_min

    for frame_id, kpts in enumerate(pos_tmp):
        for kpt_id, (x, y) in enumerate(kpts):
            s = saturation[frame_id][kpt_id]
            v = brightness[frame_id][kpt_id]
            if kpt_id in analysed_kpts_left:
                h = hue[frame_id]
            # elif kpt_id in analysed_kpts_right:
            else:
                h = 1 - hue[frame_id]
            rgb = colorsys.hsv_to_rgb(h, s, v)
            if kpt_id in all_analysed_kpts:
                img[int(y), int(x)] = rgb
                # draw_circle(img, int(x), int(y), rgb, image_width, image_height)
    return img


def main():
    if not os.path.exists('results'):
        os.mkdir('results')
    p_path = '/home/kuba/workspace/human_action_recognition/HumanActionRecognition/datasets/berkeley_mhad/3d/s01/a05/r01/3d_coordinates.npy'
    image_width = 640
    image_height = 480
    show_results = True
    save_results = True
    rotate_y = 30
    rotate_x = 0

    positions_1 = np.array(
        [np.array([rotate(k, math.radians(rotate_y), math.radians(rotate_x)) for k in f]) for f in np.load(p_path)])
    positions_x_1 = (positions_1[:, :, 0] + 1) * image_width / 2
    positions_y_1 = (positions_1[:, :, 1] + 1) * image_height / 2
    res1 = jtm_res_to_PIL_img(jtm(positions_x_1, positions_y_1, image_width, image_height))
    if show_results or save_results:
        show_results_jtm(res1, 'results/img_x_y', show_results=show_results, save_img=save_results, resize=(256, 256))

    positions_2 = np.array(
        [np.array([rotate(k, math.radians(rotate_y), math.radians(rotate_x)) for k in f]) for f in np.load(p_path)])
    positions_x_2 = (positions_2[:, :, 0] + 1) * image_width / 2
    positions_z_2 = (positions_2[:, :, 2] + 1) * image_height / 2
    res2 = jtm_res_to_PIL_img(jtm(positions_x_2, positions_z_2, image_width, image_height))
    if show_results or save_results:
        show_results_jtm(res2, 'results/img_x_z', show_results=show_results, save_img=save_results, resize=(256, 256))

    positions_3 = np.array(
        [np.array([rotate(k, math.radians(rotate_y), math.radians(rotate_x)) for k in f]) for f in np.load(p_path)])
    positions_y_3 = (positions_3[:, :, 1] + 1) * image_height / 2
    positions_z_3 = (positions_3[:, :, 2] + 1) * image_width / 2
    res3 = jtm_res_to_PIL_img(jtm(positions_z_3, positions_y_3, image_width, image_height))
    if show_results or save_results:
        show_results_jtm(res3, 'results/img_z_y', show_results=show_results, save_img=save_results, resize=(256, 256))


def get_mini_batch(shilouetes_berkeley_path, classes, image_width, image_height, samples_count=256, sample_size=(256, 256)):
    # positions = np.array([np.array([rotate(k, math.radians(rotate_y), math.radians(rotate_x)) for k in f]) for f in np.load(p_path)])
    shilouetes_dirs = sorted(
        [os.path.join(shilouetes_berkeley_path, x.name) for x in Path(shilouetes_berkeley_path).iterdir() if x.is_dir()])
    shilouetes_count = len(shilouetes_dirs)
    actions_count = len(classes)
    coordinates_file_name = '3d_coordinates.npy'
    rotations_degree_x = [0, 15, 30, 45]
    rotations_degree_y = [-45, -30, -15, 0, 15, 30, 45]
    rotations_degree_x_len = len(rotations_degree_x)
    rotations_degree_y_len = len(rotations_degree_y)

    data = []
    labels = []
    for i in range(samples_count):
        rand_shilouete_id = randrange(shilouetes_count)
        rand_action_id = randrange(actions_count)
        rand_repetition_id = randrange(5)
        coordinates_path = os.path.join(shilouetes_dirs[rand_shilouete_id],
                                        'a' + str(rand_action_id + 1).zfill(2),
                                        'r' + str(rand_repetition_id + 1).zfill(2),
                                        coordinates_file_name)
        rotation_x = math.radians(rotations_degree_x[randrange(rotations_degree_x_len)])
        rotation_y = math.radians(rotations_degree_y[randrange(rotations_degree_y_len)])
        pos = np.array([np.array([rotate(k, rotation_y, rotation_x) for k in f]) for f in np.load(coordinates_path)])
        pos_x = (pos[:, :, 0] + 1) * image_width / 2
        pos_y = (pos[:, :, 1] + 1) * image_height / 2
        sample_img = jtm_res_to_PIL_img(jtm(pos_x, pos_y, image_width, image_height)).resize(sample_size)
        # show_results_jtm(sample_img, 'results/out_{}'.format(str(i)), show_results=True, save_img=True)
        data.append(np.array(sample_img))
        labels.append(rand_action_id)
    return data, labels


if __name__ == '__main__':
    main()
    # image_width = 640
    # image_height = 480
    #
    # # Class labels
    # classes = [
    #     "JUMPING_IN_PLACE",
    #     "JUMPING_JACKS",
    #     "BENDING_HANDS_UP_ALL_THE_WAY_DOWN",
    #     "PUNCHING_BOXING",
    #     "WAVING_TWO_HANDS",
    #     "WAVING_ONE_HAND_RIGHT",
    #     "CLAPPING_HANDS",
    #     "THROWING_A_BALL",
    #     "SIT_DOWN_THEN_STAND_UP",
    #     "SIT_DOWN",
    #     "STAND_UP"
    # ]
    #
    # shilouetes_berkeley_path = '/home/kuba/workspace/human_action_recognition/HumanActionRecognition/datasets/berkeley_mhad/3d'
    # t1 = time.time()
    # data, labels = get_mini_batch(shilouetes_berkeley_path, classes, image_width, image_height)
    # t2 = time.time()
    # print(t2 - t1)
