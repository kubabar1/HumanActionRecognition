import math
import numpy as np
from PIL import Image
from matplotlib import cm

image_width = 640
image_height = 480
h_min = 0
h_max = 1

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


def to_hue(bilinear_jdm_matrix, h_minimum, h_maximum):
    return bilinear_jdm_matrix / np.max(bilinear_jdm_matrix) * (h_maximum - h_minimum)


def bilinear_resize(image, height, width):
    img_height, img_width = image.shape[:2]

    resized = np.empty([height, width])

    x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
    y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

    for i in range(height):
        for j in range(width):
            x_l, y_l = math.floor(x_ratio * j), math.floor(y_ratio * i)
            x_h, y_h = math.ceil(x_ratio * j), math.ceil(y_ratio * i)
            # x1, y1 = (0, row_id)
            # x2, y2 = (np.shape(jdm_matrix)[1]-1, np.shape(jdm_matrix)[0]-1)

            x_l = math.floor(x_ratio * j)
            x_h = math.ceil(x_ratio * j)

            x_l = x_l if x_l < img_width else img_width - 1
            x_h = x_h if x_h < img_width else img_width - 1
            y_l = y_l if y_l < img_height else img_height - 1
            y_h = y_h if y_h < img_height else img_height - 1

            x_weight = (x_ratio * j) - x_l
            y_weight = (y_ratio * i) - y_l

            a = image[y_l, x_l]
            b = image[y_l, x_h]
            c = image[y_h, x_l]
            d = image[y_h, x_h]

            resized[i][j] = a * (1 - x_weight) * (1 - y_weight) + b * x_weight * (1 - y_weight) + c * y_weight * (
                    1 - x_weight) + d * x_weight * y_weight

    return resized


def to_PIL_img(res):
    res_tmp = np.array(res)
    return Image.fromarray(np.uint8(cm.jet(res_tmp) * 255)[:, :, :3])
