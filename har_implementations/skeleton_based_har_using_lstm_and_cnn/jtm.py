import numpy as np
from scipy.spatial import distance
import colorsys
import os
import math
from utils import draw_circle, show_results_jtm, rotate


def jtm(positions_w, positions_h, image_width, image_height, L=1, s_min=0, s_max=1, b_min=0, b_max=1):
    img = np.zeros((image_height, image_width, 3))
    frames_count = len(positions_w)
    kpts_count = len(positions_w[0])
    analysed_kpts_left = [4, 5, 6, 11, 12, 13]
    analysed_kpts_right = [1, 2, 3, 14, 15, 16]
    all_analysed_kpts = analysed_kpts_left + analysed_kpts_right
    hue = np.empty([frames_count])
    v = np.empty([frames_count, kpts_count])

    for frame_id, (kpt_x, kpt_y) in enumerate(zip(positions_w, positions_h)):
        hue[frame_id] = (frame_id / (frames_count - 1)) * L
        for kpt_id, _ in enumerate(zip(kpt_x, kpt_y)):
            if frame_id < frames_count - 1:
                start = (positions_h[frame_id + 1][kpt_id], positions_w[frame_id + 1][kpt_id])
                end = (positions_h[frame_id][kpt_id], positions_w[frame_id][kpt_id])
            else:
                start = (positions_w[frame_id][kpt_id], positions_h[frame_id][kpt_id])
                end = (positions_w[frame_id - 1][kpt_id], positions_h[frame_id - 1][kpt_id])
            v[frame_id][kpt_id] = distance.euclidean(start, end)

    saturation = v / np.max(v) * (s_max - s_min) + s_min
    brightness = v / np.max(v) * (b_max - b_min) + b_min

    for frame_id, (kpt_x, kpt_y) in enumerate(zip(positions_w, positions_h)):
        for kpt_id, (x, y) in enumerate(zip(kpt_x, kpt_y)):
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
    res1 = jtm(positions_x_1, positions_y_1, image_width, image_height)
    if show_results or save_results:
        show_results_jtm(res1, 'results/img_x_y', show_results=show_results, save_img=save_results)

    positions_2 = np.array(
        [np.array([rotate(k, math.radians(rotate_y), math.radians(rotate_x)) for k in f]) for f in np.load(p_path)])
    positions_x_2 = (positions_2[:, :, 0] + 1) * image_width / 2
    positions_z_2 = (positions_2[:, :, 2] + 1) * image_height / 2
    res2 = jtm(positions_x_2, positions_z_2, image_width, image_height)
    if show_results or save_results:
        show_results_jtm(res2, 'results/img_x_z', show_results=show_results, save_img=save_results)

    positions_3 = np.array(
        [np.array([rotate(k, math.radians(rotate_y), math.radians(rotate_x)) for k in f]) for f in np.load(p_path)])
    positions_y_3 = (positions_3[:, :, 1] + 1) * image_height / 2
    positions_z_3 = (positions_3[:, :, 2] + 1) * image_width / 2
    res3 = jtm(positions_z_3, positions_y_3, image_width, image_height)
    if show_results or save_results:
        show_results_jtm(res3, 'results/img_z_y', show_results=show_results, save_img=save_results)


if __name__ == '__main__':
    main()
