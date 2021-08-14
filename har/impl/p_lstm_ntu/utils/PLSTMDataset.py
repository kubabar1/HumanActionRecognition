from random import randrange

import numpy as np
from torch.utils.data import Dataset

from har.utils.dataset_util import video_pose_3d_kpts


class PLSTMDataset(Dataset):
    def __init__(self, data, labels, batch_size, split=20):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        split = self.split

        right_wrists = []
        left_wrists = []
        right_elbows = []
        left_elbows = []
        right_shoulders = []
        left_shoulders = []
        right_hips = []
        left_hips = []
        right_knees = []
        left_knees = []
        right_ankles = []
        left_ankles = []

        labels_arr = []

        while len(labels_arr) < self.batch_size:
            random_data_idx = randrange(self.__len__())
            data_el = self.data[random_data_idx]
            label_el = self.labels[random_data_idx]

            right_wrists.append(np.array(
                [a[randrange(len(a))] for a in np.array_split(data_el[:, video_pose_3d_kpts['right_wrist'], :], split)]))
            left_wrists.append(np.array(
                [a[randrange(len(a))] for a in np.array_split(data_el[:, video_pose_3d_kpts['left_wrist'], :], split)]))
            right_elbows.append(np.array(
                [a[randrange(len(a))] for a in np.array_split(data_el[:, video_pose_3d_kpts['right_elbow'], :], split)]))
            left_elbows.append(np.array(
                [a[randrange(len(a))] for a in np.array_split(data_el[:, video_pose_3d_kpts['left_elbow'], :], split)]))
            right_shoulders.append(np.array(
                [a[randrange(len(a))] for a in np.array_split(data_el[:, video_pose_3d_kpts['right_shoulder'], :], split)]))
            left_shoulders.append(np.array(
                [a[randrange(len(a))] for a in np.array_split(data_el[:, video_pose_3d_kpts['left_shoulder'], :], split)]))
            right_hips.append(
                np.array([a[randrange(len(a))] for a in np.array_split(data_el[:, video_pose_3d_kpts['right_hip'], :], split)]))
            left_hips.append(
                np.array([a[randrange(len(a))] for a in np.array_split(data_el[:, video_pose_3d_kpts['left_hip'], :], split)]))
            right_knees.append(np.array(
                [a[randrange(len(a))] for a in np.array_split(data_el[:, video_pose_3d_kpts['right_knee'], :], split)]))
            left_knees.append(
                np.array([a[randrange(len(a))] for a in np.array_split(data_el[:, video_pose_3d_kpts['left_knee'], :], split)]))
            right_ankles.append(np.array(
                [a[randrange(len(a))] for a in np.array_split(data_el[:, video_pose_3d_kpts['right_ankle'], :], split)]))
            left_ankles.append(np.array(
                [a[randrange(len(a))] for a in np.array_split(data_el[:, video_pose_3d_kpts['left_ankle'], :], split)]))

            labels_arr.append(label_el)
            if len(labels_arr) >= self.batch_size:
                break

        left_arms = np.concatenate((left_wrists, left_elbows, left_shoulders), axis=2)
        right_arms = np.concatenate((right_wrists, right_elbows, right_shoulders), axis=2)
        left_legs = np.concatenate((left_hips, left_knees, left_ankles), axis=2)
        right_legs = np.concatenate((right_hips, right_knees, right_ankles), axis=2)

        return [left_arms, right_arms, left_legs, right_legs], np.array(labels_arr)
