import colorsys
from math import cos, sin

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import distance


def jtm(positions_w, positions_h, image_width, image_height, analysed_kpts_left, analysed_kpts_right, L=1, s_min=0, s_max=1,
        b_min=0, b_max=1):
    img = np.zeros((image_height, image_width, 3))
    frames_count = len(positions_w)
    kpts_count = len(positions_w[0])
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
            else:
                h = 1 - hue[frame_id]
            rgb = colorsys.hsv_to_rgb(h, s, v)
            # img[int(y), int(x)] = rgb
            draw_circle(img, int(x), int(y), rgb, image_width, image_height)
    return img


def jtm_res_to_pil_img(res):
    res_tmp = np.array(res)
    res_tmp *= 255
    res_tmp = np.array(res_tmp, dtype=np.uint8)
    # res_tmp[res_tmp == 0] = 255
    return Image.fromarray(res_tmp, 'RGB')


def show_results_jtm(res, title, show_results=True, save_img=False, resize=None):
    # eps = np.spacing(0.0)
    # im1 = plt.pcolormesh(res, cmap=plt.cm.jet, vmin=eps)
    # plt.imshow(res, cmap=plt.cm.jet, vmin=eps)
    res_tmp = np.array(res)
    if show_results:
        fig = plt.gcf()
        fig.canvas.set_window_title(title)
        plt.imshow(res_tmp, cmap=plt.cm.jet)
        plt.axis('off')
        plt.show()
    if save_img:
        img = Image.fromarray(res_tmp, 'RGB')
        if resize:
            img = img.resize(resize)
        img.save('{}.png'.format(title))
    plt.close()


def draw_circle(img, x, y, rgb, image_width, image_height):
    for i in range(5):
        x_c = int(x) - 2 + i
        for j in range(5):
            y_c = int(y) - 2 + j
            if 0 < y_c < image_height and 0 < x_c < image_width:
                img[y_c, x_c] = rgb

    if 0 < y - 1 < image_height and 0 < x - 3 < image_width:
        img[y - 1, x - 3] = rgb
    if 0 < y < image_height and 0 < x - 3 < image_width:
        img[y, x - 3] = rgb
    if 0 < y + 1 < image_height and 0 < x - 3 < image_width:
        img[y + 1, x - 3] = rgb
    if 0 < y - 1 < image_height and 0 < x + 3 < image_width:
        img[y - 1, x + 3] = rgb
    if 0 < y < image_height and 0 < x + 3 < image_width:
        img[y, x + 3] = rgb
    if 0 < y + 1 < image_height and 0 < x + 3 < image_width:
        img[y + 1, x + 3] = rgb

    if 0 < y - 3 < image_height and 0 < x - 1 < image_width:
        img[y - 3, x - 1] = rgb
    if 0 < y - 3 < image_height and 0 < x < image_width:
        img[y - 3, x] = rgb
    if 0 < y - 3 < image_height and 0 < x + 1 < image_width:
        img[y - 3, x + 1] = rgb
    if 0 < y - 3 < image_height and 0 < x - 1 < image_width:
        img[y - 3, x - 1] = rgb
    if 0 < y - 3 < image_height and 0 < x < image_width:
        img[y - 3, x] = rgb
    if 0 < y - 3 < image_height and 0 < x + 1 < image_width:
        img[y - 3, x + 1] = rgb


def rotate(coordinates, rotate_y, rotate_x):
    _, _, z = coordinates
    return np.array(np.matmul(np.matmul(t_r_y(rotate_y, z), t_r_x(rotate_x, z)), np.append(coordinates, 1).transpose())[:3])


def t_r_y(phi, z):
    return np.array(
        [
            [1, 0, 0, 0],
            [0, cos(phi), -sin(phi), z * sin(phi)],
            [0, sin(phi), cos(phi), z * (1 - cos(phi))],
            [0, 0, 0, 1]
        ]
    )


def t_r_x(theta, z):
    return np.array(
        [
            [cos(theta), 0, sin(theta), -z * sin(theta)],
            [0, 1, 0, 0],
            [-sin(theta), 0, cos(theta), z * (1 - cos(theta))],
            [0, 0, 0, 1]
        ]
    )
