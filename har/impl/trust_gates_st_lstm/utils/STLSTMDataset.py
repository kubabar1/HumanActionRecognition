from random import randrange

import numpy as np
from torch.utils.data import Dataset

from har.utils.dataset_util import get_analysed_keypoints


class STLSTMDataset(Dataset):
    def __init__(self, data, labels, batch_size, split=20):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_arr = []
        labels_arr = []

        while len(data_arr) < self.batch_size:
            random_data_idx = randrange(self.__len__())
            data_el = self.data[random_data_idx]
            label_el = self.labels[random_data_idx]
            # print( .shape)
            analysed_kpts_left, analysed_kpts_right = get_analysed_keypoints()
            all_analysed_kpts = analysed_kpts_left + analysed_kpts_right

            data_el_split = np.array([a[randrange(len(a)), all_analysed_kpts, :] for a in np.array_split(data_el, self.split)])

            data_arr.append(data_el_split)
            labels_arr.append(label_el)
        return np.array(data_arr), np.array(labels_arr)

# def get_data2(dataset_path, batch_size, analysed_kpts):
#     data, labels = get_batch(dataset_path, batch_size=batch_size, training=True)
#     shoulders_m = (data[:, :, video_pose_3d_kpts['left_shoulder'], :] + data[:, :, video_pose_3d_kpts['right_shoulder'], :]) / 2
#     hip_m = (data[:, :, video_pose_3d_kpts['left_hip'], :] + data[:, :, video_pose_3d_kpts['right_hip'], :]) / 2
#     hip_shoulder_m = (shoulders_m + hip_m) / 2
#     data = np.stack([
#         hip_shoulder_m,
#         shoulders_m,
#         data[:, :, video_pose_3d_kpts['right_shoulder'], :],
#         data[:, :, video_pose_3d_kpts['right_elbow'], :],
#         data[:, :, video_pose_3d_kpts['right_wrist'], :],
#         data[:, :, video_pose_3d_kpts['right_elbow'], :],
#         data[:, :, video_pose_3d_kpts['right_shoulder'], :],
#         shoulders_m,
#         data[:, :, video_pose_3d_kpts['left_shoulder'], :],
#         data[:, :, video_pose_3d_kpts['left_elbow'], :],
#         data[:, :, video_pose_3d_kpts['left_wrist'], :],
#         data[:, :, video_pose_3d_kpts['left_elbow'], :],
#         data[:, :, video_pose_3d_kpts['left_shoulder'], :],
#         shoulders_m,
#         hip_shoulder_m,
#         hip_m,
#         data[:, :, video_pose_3d_kpts['right_hip'], :],
#         data[:, :, video_pose_3d_kpts['right_knee'], :],
#         data[:, :, video_pose_3d_kpts['right_ankle'], :],
#         data[:, :, video_pose_3d_kpts['right_knee'], :],
#         data[:, :, video_pose_3d_kpts['right_hip'], :],
#         hip_m,
#         data[:, :, video_pose_3d_kpts['left_hip'], :],
#         data[:, :, video_pose_3d_kpts['left_knee'], :],
#         data[:, :, video_pose_3d_kpts['left_ankle'], :],
#         data[:, :, video_pose_3d_kpts['left_knee'], :],
#         data[:, :, video_pose_3d_kpts['left_hip'], :],
#         hip_m,
#         hip_shoulder_m
#     ], axis=2)
#     return np.transpose(data, (
#         1, 2, 0, 3)), labels  # [frame, joint, batch, channels]def get_data(dataset_path, batch_size, analysed_kpts):
#     data, labels = get_batch(dataset_path, batch_size=batch_size, training=True)
#     shoulders_m = (data[:, :, video_pose_3d_kpts['left_shoulder'], :] + data[:, :, video_pose_3d_kpts['right_shoulder'], :]) / 2
#     hip_m = (data[:, :, video_pose_3d_kpts['left_hip'], :] + data[:, :, video_pose_3d_kpts['right_hip'], :]) / 2
#     hip_shoulder_m = (shoulders_m + hip_m) / 2
#     data = np.stack([
#         hip_shoulder_m,
#         shoulders_m,
#         data[:, :, video_pose_3d_kpts['right_shoulder'], :],
#         data[:, :, video_pose_3d_kpts['right_elbow'], :],
#         data[:, :, video_pose_3d_kpts['right_wrist'], :],
#         data[:, :, video_pose_3d_kpts['right_elbow'], :],
#         data[:, :, video_pose_3d_kpts['right_shoulder'], :],
#         shoulders_m,
#         data[:, :, video_pose_3d_kpts['left_shoulder'], :],
#         data[:, :, video_pose_3d_kpts['left_elbow'], :],
#         data[:, :, video_pose_3d_kpts['left_wrist'], :],
#         data[:, :, video_pose_3d_kpts['left_elbow'], :],
#         data[:, :, video_pose_3d_kpts['left_shoulder'], :],
#         shoulders_m,
#         hip_shoulder_m,
#         hip_m,
#         data[:, :, video_pose_3d_kpts['right_hip'], :],
#         data[:, :, video_pose_3d_kpts['right_knee'], :],
#         data[:, :, video_pose_3d_kpts['right_ankle'], :],
#         data[:, :, video_pose_3d_kpts['right_knee'], :],
#         data[:, :, video_pose_3d_kpts['right_hip'], :],
#         hip_m,
#         data[:, :, video_pose_3d_kpts['left_hip'], :],
#         data[:, :, video_pose_3d_kpts['left_knee'], :],
#         data[:, :, video_pose_3d_kpts['left_ankle'], :],
#         data[:, :, video_pose_3d_kpts['left_knee'], :],
#         data[:, :, video_pose_3d_kpts['left_hip'], :],
#         hip_m,
#         hip_shoulder_m
#     ], axis=2)
#     return np.transpose(data, (1, 2, 0, 3)), labels  # [frame, joint, batch, channels]
