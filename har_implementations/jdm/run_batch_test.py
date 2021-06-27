import os
import matplotlib.pyplot as plt
import numpy as np
from impl.jdm import jdm, bilinear_resize, get_mini_batch
from impl.utils import to_hue, h_min, h_max, classes

dataset_path = '../../datasets/berkeley_mhad/3d'

positions = (np.array(np.load(os.path.join(dataset_path, 's01/a01/r01/3d_coordinates.npy'))))
channel1, channel2, channel3, channel4 = jdm(positions)

analysed_channel = channel1

print(analysed_channel.shape)

h = to_hue(analysed_channel, h_min, h_max)

plt.imshow(h, cmap=plt.cm.jet)
plt.show()

i = 1
# positions = np.stack([positions_x_1, positions_y_1], axis=2)
positions = (np.array(np.load(os.path.join(dataset_path, 's01/a{}/r01/3d_coordinates.npy').format(str(i + 1).zfill(2)))) + 1)
channel1, channel2, channel3, channel4 = jdm(positions)

analysed_channel = channel1

print(analysed_channel.shape)

b = bilinear_resize(np.array(analysed_channel), analysed_channel.shape[0], 200)
h = to_hue(b, h_min, h_max)

plt.imshow(h, cmap=plt.cm.jet)
plt.show()
print('Resized channel')
print(classes[i])
print(h.shape)
print(np.min(h))
print(np.max(h))
print(np.mean(h))
print(np.std(h))

positions = (np.array(np.load(os.path.join(dataset_path, 's01/a{}/r01/3d_coordinates.npy').format(str(i + 1).zfill(2)))) + 1)
channel1, channel2, channel3, channel4 = jdm(positions)

analysed_channel = channel1
b = bilinear_resize(np.array(analysed_channel), analysed_channel.shape[0], 200)
h = to_hue(b, h_min, h_max)

plt.imshow(h, cmap=plt.cm.jet)
plt.show()
print('Resized channel')
print(classes[i])
print(h.shape)
print(np.min(h))
print(np.max(h))
print(np.mean(h))
print(np.std(h))

positions = (np.array(np.load(os.path.join(dataset_path, 's01/a{}/r01/3d_coordinates.npy').format(str(i + 1).zfill(2)))) + 1)
channel1, channel2, channel3, channel4 = jdm(positions)

analysed_channel = channel1
b = bilinear_resize(np.array(analysed_channel), analysed_channel.shape[0], 200)
h = to_hue(b, h_min, h_max)

plt.imshow(h, cmap=plt.cm.jet)
plt.show()
print('Resized channel')
print(classes[i])
print(h.shape)
print(np.min(h))
print(np.max(h))
print(np.mean(h))
print(np.std(h))

positions = (np.array(np.load(os.path.join(dataset_path, 's01/a{}/r01/3d_coordinates.npy').format(str(i + 1).zfill(2)))) + 1)
channel1, channel2, channel3, channel4 = jdm(positions)

analysed_channel = channel1
b = bilinear_resize(np.array(analysed_channel), analysed_channel.shape[0], 200)
h = to_hue(b, h_min, h_max)

plt.imshow(h, cmap=plt.cm.jet)
plt.show()
print('Resized channel')
print(classes[i])
print(h.shape)
print(np.min(h))
print(np.max(h))
print(np.mean(h))
print(np.std(h))

jdm_images, labels = get_mini_batch(dataset_path, classes, samples_count=3, training=True)

print(len(jdm_images))
print(len(labels))
print(jdm_images[0][0])
print(labels)
print(np.array(jdm_images[0][0]).shape)
# print(np.array(jdm_images[0][0]))
print(classes[labels[0]])
plt.imshow(jdm_images[0][0])
plt.show()
# plt.imshow(jdm_images[0][1])
# plt.show()
# plt.imshow(jdm_images[0][2])
# plt.show()
# plt.imshow(jdm_images[0][3])
# plt.show()
print(classes[labels[1]])
plt.imshow(jdm_images[1][0])
plt.show()
# plt.imshow(jdm_images[1][1])
# plt.show()
# plt.imshow(jdm_images[1][2])
# plt.show()
# plt.imshow(jdm_images[1][3])
# plt.show()
print(classes[labels[2]])
plt.imshow(jdm_images[2][0])
plt.show()
