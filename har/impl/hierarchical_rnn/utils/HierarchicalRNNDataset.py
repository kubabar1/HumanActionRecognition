from random import randrange

import numpy as np
from torch.utils.data import Dataset

from ....utils.dataset_util import DatasetInputType, random_rotate_y, get_all_body_parts_steps, get_all_body_parts_splits


class HierarchicalRNNDataset(Dataset):
    def __init__(self, data, labels, batch_size, analysed_kpts_description, add_random_rotation_y=False, steps=32, split=20,
                 input_type=DatasetInputType.SPLIT, is_test=False):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.split = split
        self.analysed_kpts_description = analysed_kpts_description
        self.add_random_rotation_y = add_random_rotation_y
        self.input_type = input_type
        self.steps = steps
        self.split = split
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        analysed_body_parts = ['right_wrist', 'left_wrist', 'right_elbow', 'left_elbow', 'right_shoulder', 'left_shoulder',
                               'right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_ankle', 'left_ankle']
        data_len = len(self.data)

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

        it = 0

        while it < self.batch_size:
            if self.is_test:
                data_el = self.data[it % data_len]
                label_el = self.labels[it % data_len]
            else:
                random_data_idx = randrange(self.__len__())
                data_el = self.data[random_data_idx]
                label_el = self.labels[random_data_idx]

            if self.add_random_rotation_y:
                data_el = random_rotate_y(data_el)

            if self.input_type == DatasetInputType.STEP:
                parts = int(data_el.shape[0] / self.steps)
                for i in range(parts):
                    begin = i * self.steps
                    end = i * self.steps + self.steps
                    body_el = get_all_body_parts_steps(data_el, analysed_body_parts, self.analysed_kpts_description, begin, end)
                    right_wrists.append(body_el['right_wrist'])
                    left_wrists.append(body_el['left_wrist'])
                    right_elbows.append(body_el['right_elbow'])
                    left_elbows.append(body_el['left_elbow'])
                    right_shoulders.append(body_el['right_shoulder'])
                    left_shoulders.append(body_el['left_shoulder'])
                    right_hips.append(body_el['right_hip'])
                    left_hips.append(body_el['left_hip'])
                    right_knees.append(body_el['right_knee'])
                    left_knees.append(body_el['left_knee'])
                    right_ankles.append(body_el['right_ankle'])
                    left_ankles.append(body_el['left_ankle'])
                    if len(left_ankles) >= self.batch_size:
                        break
            elif self.input_type == DatasetInputType.SPLIT:
                body_el = get_all_body_parts_splits(data_el, analysed_body_parts, self.analysed_kpts_description, self.split)
                right_wrists.append(body_el['right_wrist'])
                left_wrists.append(body_el['left_wrist'])
                right_elbows.append(body_el['right_elbow'])
                left_elbows.append(body_el['left_elbow'])
                right_shoulders.append(body_el['right_shoulder'])
                left_shoulders.append(body_el['left_shoulder'])
                right_hips.append(body_el['right_hip'])
                left_hips.append(body_el['left_hip'])
                right_knees.append(body_el['right_knee'])
                left_knees.append(body_el['left_knee'])
                right_ankles.append(body_el['right_ankle'])
                left_ankles.append(body_el['left_ankle'])
            else:
                raise ValueError('Invalid or unimplemented input type')

            labels_arr.append(label_el)
            if len(labels_arr) >= self.batch_size:
                break
            it += 1

        left_arms = np.concatenate((left_wrists, left_elbows, left_shoulders), axis=2)
        right_arms = np.concatenate((right_wrists, right_elbows, right_shoulders), axis=2)
        left_legs = np.concatenate((left_hips, left_knees, left_ankles), axis=2)
        right_legs = np.concatenate((right_hips, right_knees, right_ankles), axis=2)

        return [left_arms, right_arms, left_legs, right_legs], np.array(labels_arr)
