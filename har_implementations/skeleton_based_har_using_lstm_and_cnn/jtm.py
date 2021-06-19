import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import colorsys
from PIL import Image


def jtm(positions_w, positions_h, image_width, image_height, L=1, s_min=0, s_max=1, b_min=0, b_max=1):
    img = np.zeros((image_height, image_width, 3))
    frames_count = len(positions_w)
    kpts_count = len(positions_w[0])
    analysed_kpts_left = [4, 5, 6, 11, 12, 13]
    analysed_kpts_right = [1, 2, 3, 14, 15, 16]
    hue = np.empty([frames_count])
    v = np.empty([frames_count, kpts_count])

    for frame_id, (kpt_x, kpt_y) in enumerate(zip(positions_w, positions_h)):
        hue[frame_id] = (frame_id / (frames_count - 1)) * L
        for kpt_id, _ in enumerate(zip(kpt_x, kpt_y)):
            if frame_id < frames_count - 1:
                start = (positions_w[frame_id + 1][kpt_id], positions_w[frame_id + 1][kpt_id])
                end = (positions_h[frame_id][kpt_id], positions_h[frame_id][kpt_id])
            else:
                start = (positions_w[frame_id][kpt_id], positions_w[frame_id][kpt_id])
                end = (positions_h[frame_id - 1][kpt_id], positions_h[frame_id - 1][kpt_id])
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
            img[int(y), int(x)] = rgb
    return img


def show_results(res, title, save_img=False):
    # eps = np.spacing(0.0)
    # im1 = plt.pcolormesh(res, cmap=plt.cm.jet, vmin=eps)
    # plt.imshow(res, cmap=plt.cm.jet, vmin=eps)
    # res[res == 0] = 255
    plt.imshow(res, cmap=plt.cm.jet)
    plt.axis('off')
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    if save_img:
        plt.savefig('{}.jpg'.format(title), bbox_inches='tight')
    plt.show()
    plt.close()



def main():
    p_path = '/home/kuba/workspace/human_action_recognition/HumanActionRecognition/datasets/berkeley_mhad/3d/s01/a05/r01/3d_coordinates.npy'
    image_width = 640
    image_height = 480

    positions_1 = np.load(p_path)
    positions_x_1 = (positions_1[:, :, 0] + 1) * image_width / 2
    positions_y_1 = (positions_1[:, :, 1] + 1) * image_height / 2
    res1 = jtm(positions_x_1, positions_y_1, image_width, image_height)
    res1[res1 == 0] = 1
    show_results(res1, 'img_x_y', save_img=True)
    res1 *= 255
    img = Image.fromarray(np.array(res1, dtype=np.uint8), 'RGB')
    img.save('out1.png')

    positions_2 = np.load(p_path)
    positions_x_2 = (positions_2[:, :, 0] + 1) * image_width / 2
    positions_z_2 = (positions_2[:, :, 2] + 1) * image_height / 2
    res2 = jtm(positions_x_2, positions_z_2, image_width, image_height)
    res2[res2 == 0] = 1
    show_results(res2, 'img_x_z', save_img=True)
    res2 *= 255
    img = Image.fromarray(np.array(res2, dtype=np.uint8), 'RGB')
    img.save('out2.png')

    positions_3 = np.load(p_path)
    positions_y_3 = (positions_3[:, :, 1] + 1) * image_height / 2
    positions_z_3 = (positions_3[:, :, 2] + 1) * image_width / 2
    res3 = jtm(positions_z_3, positions_y_3, image_width, image_height)
    res3[res3 == 0] = 1
    show_results(res3, 'img_z_y', save_img=True)
    res3 *= 255
    img = Image.fromarray(np.array(res3, dtype=np.uint8), 'RGB')
    img.save('out3.png')


if __name__ == '__main__':
    main()
