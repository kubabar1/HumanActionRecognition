import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import colorsys
import os
from PIL import Image


def draw_circle(img, x, y, rgb, image_width, image_height):
    for i in range(5):
        x_c = int(x) - 2 + i
        for j in range(5):
            y_c = int(y) - 2 + j
            if 0 < y_c < image_height and 0 < x_c < image_width:
                img[y_c, x_c] = rgb
    img[y - 1, x - 3] = rgb
    img[y, x - 3] = rgb
    img[y + 1, x - 3] = rgb
    img[y - 1, x + 3] = rgb
    img[y, x + 3] = rgb
    img[y + 1, x + 3] = rgb

    img[y - 3, x - 1] = rgb
    img[y - 3, x] = rgb
    img[y - 3, x + 1] = rgb
    img[y - 3, x - 1] = rgb
    img[y - 3, x] = rgb
    img[y - 3, x + 1] = rgb


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
                draw_circle(img, int(x), int(y), rgb, image_width, image_height)
    return img


def show_results(res, title, save_img=False):
    # eps = np.spacing(0.0)
    # im1 = plt.pcolormesh(res, cmap=plt.cm.jet, vmin=eps)
    # plt.imshow(res, cmap=plt.cm.jet, vmin=eps)
    res_tmp = res.copy()
    res_tmp[res_tmp == 0] = 1
    plt.imshow(res_tmp, cmap=plt.cm.jet)
    plt.axis('off')
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    if save_img:
        res_tmp *= 255
        img = Image.fromarray(np.array(res_tmp, dtype=np.uint8), 'RGB')
        img.save('{}.png'.format(title))
    plt.show()
    plt.close()


def main():
    if not os.path.exists('results'):
        os.mkdir('results')
    p_path = '/home/kuba/workspace/human_action_recognition/HumanActionRecognition/datasets/berkeley_mhad/3d/s01/a05/r01/3d_coordinates.npy'
    image_width = 640
    image_height = 480

    positions_1 = np.load(p_path)
    positions_x_1 = (positions_1[:, :, 0] + 1) * image_width / 2
    positions_y_1 = (positions_1[:, :, 1] + 1) * image_height / 2
    res1 = jtm(positions_x_1, positions_y_1, image_width, image_height)
    show_results(res1, 'results/img_x_y', save_img=True)

    positions_2 = np.load(p_path)
    positions_x_2 = (positions_2[:, :, 0] + 1) * image_width / 2
    positions_z_2 = (positions_2[:, :, 2] + 1) * image_height / 2
    res2 = jtm(positions_x_2, positions_z_2, image_width, image_height)
    show_results(res2, 'results/img_x_z', save_img=True)

    positions_3 = np.load(p_path)
    positions_y_3 = (positions_3[:, :, 1] + 1) * image_height / 2
    positions_z_3 = (positions_3[:, :, 2] + 1) * image_width / 2
    res3 = jtm(positions_z_3, positions_y_3, image_width, image_height)
    show_results(res3, 'results/img_z_y', save_img=True)


if __name__ == '__main__':
    main()
