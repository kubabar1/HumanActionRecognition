import os
from pathlib import Path
from random import randrange

import numpy as np

# def get_batch(shilouetes_berkeley_path, batch_size=128, training=True, train_threshold=0.8, shilouetes_count=8,
#               actions_count=11, repetitions_count=5, split_t=20):
#     shilouetes_dirs = sorted(
#         [os.path.join(shilouetes_berkeley_path, x.name) for x in Path(shilouetes_berkeley_path).iterdir() if x.is_dir()])
#     coordinates_file_name = '3d_coordinates.npy'
#     train_size = int(shilouetes_count * train_threshold)
#     train_shilouetes = range(train_size)
#     test_shilouetes = range(shilouetes_count - train_size)
#
#     data = []
#     labels = []
#     for i in range(batch_size):
#         rand_shilouete_id = randrange(shilouetes_count)
#         rand_action_id = randrange(actions_count)
#         rand_repetition_id = randrange(repetitions_count)
#         coordinates_path = os.path.join(shilouetes_dirs[rand_shilouete_id],
#                                         'a' + str(rand_action_id + 1).zfill(2),
#                                         'r' + str(rand_repetition_id + 1).zfill(2),
#                                         coordinates_file_name)
#         data.append(np.array([a[randrange(len(a))] for a in np.array_split(np.load(coordinates_path), split_t)]))
#
#         # if len(pos) < sequence_len:
#         #     tmp = []
#         #     rp = sequence_len // len(pos)
#         #     lt = sequence_len - len(pos) * rp
#         #     for _ in range(rp):
#         #         tmp.extend(pos)
#         #     tmp.extend(pos[:lt])
#         #     data.append(np.array(tmp))
#         # else:
#         #     data.append(pos[:sequence_len])
#         labels.append(rand_action_id)
#
#     analysed_kpts_left = [4, 5, 6, 11, 12, 13]
#     analysed_kpts_right = [1, 2, 3, 14, 15, 16]
#     all_analysed_kpts = analysed_kpts_left + analysed_kpts_right
#
#     data = np.array(data, dtype='float')
#     data = data[:, :, all_analysed_kpts, :]
#     return np.transpose(data, (1, 2, 0, 3)), labels  # [frame, joint, batch, channels]


# def get_batch(shilouetes_berkeley_path, batch_size=128, training=True, train_threshold=0.8, shilouetes_count=8,
#               actions_count=11, repetitions_count=5, split_t=20):
#     shilouetes_dirs = sorted(
#         [os.path.join(shilouetes_berkeley_path, x.name) for x in Path(shilouetes_berkeley_path).iterdir() if x.is_dir()])
#     coordinates_file_name = '3d_coordinates.npy'
#     train_size = int(shilouetes_count * train_threshold)
#     train_shilouetes = range(train_size)
#     test_shilouetes = range(shilouetes_count - train_size)
#
#     data = []
#     labels = []
#     for i in range(batch_size):
#         rand_shilouete_id = randrange(shilouetes_count)
#         rand_action_id = randrange(actions_count)
#         rand_repetition_id = randrange(repetitions_count)
#         coordinates_path = os.path.join(shilouetes_dirs[rand_shilouete_id],
#                                         'a' + str(rand_action_id + 1).zfill(2),
#                                         'r' + str(rand_repetition_id + 1).zfill(2),
#                                         coordinates_file_name)
#         data.append(np.array([a[randrange(len(a))] for a in np.array_split(np.load(coordinates_path), split_t)]))
#
#         # if len(pos) < sequence_len:
#         #     tmp = []
#         #     rp = sequence_len // len(pos)
#         #     lt = sequence_len - len(pos) * rp
#         #     for _ in range(rp):
#         #         tmp.extend(pos)
#         #     tmp.extend(pos[:lt])
#         #     data.append(np.array(tmp))
#         # else:
#         #     data.append(pos[:sequence_len])
#         labels.append(rand_action_id)
#
#     return np.array(data, dtype='float'), labels


# def get_batch2(shilouetes_berkeley_path, sequence_len=100, batch_size=128, training=True, train_threshold=0.8, shilouetes_count=8,
#                actions_count=11, repetitions_count=5):
#     shilouetes_dirs = sorted(
#         [os.path.join(shilouetes_berkeley_path, x.name) for x in Path(shilouetes_berkeley_path).iterdir() if x.is_dir()])
#     coordinates_file_name = '3d_coordinates.npy'
#     train_size = int(shilouetes_count * train_threshold)
#     train_shilouetes = range(train_size)
#     test_shilouetes = range(shilouetes_count - train_size)
#
#     data = []
#     labels = []
#     for i in range(batch_size):
#         rand_shilouete_id = randrange(shilouetes_count)
#         rand_action_id = randrange(actions_count)
#         rand_repetition_id = randrange(repetitions_count)
#         coordinates_path = os.path.join(shilouetes_dirs[rand_shilouete_id],
#                                         'a' + str(rand_action_id + 1).zfill(2),
#                                         'r' + str(rand_repetition_id + 1).zfill(2),
#                                         coordinates_file_name)
#         pos = np.array(np.load(coordinates_path))
#
#         if len(pos) < sequence_len:
#             tmp = []
#             rp = sequence_len // len(pos)
#             lt = sequence_len - len(pos) * rp
#             for _ in range(rp):
#                 tmp.extend(pos)
#             tmp.extend(pos[:lt])
#             data.append(np.array(tmp))
#         else:
#             data.append(pos[:sequence_len])
#         labels.append(rand_action_id)
#
#     return np.array(data, dtype='float'), labels
