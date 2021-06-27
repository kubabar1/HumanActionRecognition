from math import cos, sin
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

image_width = 640
image_height = 480

# Class labels
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


def jtm_res_to_PIL_img(res):
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
    return np.array(np.matmul(np.matmul(T_r_y(rotate_y, z), T_r_x(rotate_x, z)), np.append(coordinates, 1).transpose())[:3])


T_r_y = lambda phi, z: np.array(
    [
        [1, 0, 0, 0],
        [0, cos(phi), -sin(phi), z * sin(phi)],
        [0, sin(phi), cos(phi), z * (1 - cos(phi))],
        [0, 0, 0, 1]
    ]
)

T_r_x = lambda theta, z: np.array(
    [
        [cos(theta), 0, sin(theta), -z * sin(theta)],
        [0, 1, 0, 0],
        [-sin(theta), 0, cos(theta), z * (1 - cos(theta))],
        [0, 0, 0, 1]
    ]
)
