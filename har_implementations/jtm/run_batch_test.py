import os
import math
import time
import numpy as np
from impl.utils import rotate, jtm_res_to_PIL_img, show_results_jtm, image_height, image_width, classes
from matplotlib import pyplot as plt
from impl.jtm import jtm, get_mini_batch

if not os.path.exists('results'):
    os.mkdir('results')
p_path = '../../datasets/berkeley_mhad/3d/s01/a05/r01/3d_coordinates.npy'
show_results = True
save_results = True
rotate_y = 30
rotate_x = 45

positions_1 = np.array(
    [np.array([rotate(k, math.radians(rotate_y), math.radians(rotate_x)) for k in f]) for f in np.load(p_path)])
positions_1 = np.load(p_path)
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

t1 = time.time()
data, labels = get_mini_batch('../../datasets/berkeley_mhad/3d', classes, image_width, image_height, samples_count=10)
t2 = time.time()

print(len(data))
print(len(labels))
print(data[0][0])
print(labels)
print(t2 - t1)
plt.imshow(data[0][0])
plt.show()
plt.imshow(data[0][1])
plt.show()
plt.imshow(data[0][2])
plt.show()
