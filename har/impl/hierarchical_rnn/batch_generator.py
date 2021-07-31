import os
from random import randrange

import numpy as np

from har.utils.dataset_utils import video_pose_3d_kpts


def get_batch_ntu_rgbd(dataset_path, batch_size=128, steps=32, is_training=True, data_npy_file_name='3d_coordinates.npy'):
    scenes_count = 1
    cameras_count = 3
    persons_count = 8
    repetitions_count = 2

    excluded_actions = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
    all_actions = list(range(1, 61))

    for a in excluded_actions:
        all_actions.remove(a)

    scenes_id = scenes_count

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

    while len(labels_arr) < batch_size:
        camera_id = randrange(cameras_count) + 1
        person_id = randrange(6) + 1 if is_training else randrange(6, persons_count) + 1
        repetition_id = randrange(repetitions_count) + 1
        action_id = randrange(len(all_actions))

        selected_dir = 'S{}C{}P{}R{}A{}_rgb'.format(str(scenes_id).zfill(3), str(camera_id).zfill(3), str(person_id).zfill(3),
                                                    str(repetition_id).zfill(3), str(all_actions[action_id]).zfill(3))
        data_path = os.path.join(dataset_path, selected_dir, data_npy_file_name)
        data = np.array(np.load(data_path))
        label = all_actions[action_id]
        parts = int(data.shape[0] / steps)

        for i in range(parts):
            start = i * steps
            end = i * steps + steps

            right_wrists.append(data[start:end, video_pose_3d_kpts['right_wrist'], :])
            left_wrists.append(data[start:end, video_pose_3d_kpts['left_wrist'], :])
            right_elbows.append(data[start:end, video_pose_3d_kpts['right_elbow'], :])
            left_elbows.append(data[start:end, video_pose_3d_kpts['left_elbow'], :])
            right_shoulders.append(data[start:end, video_pose_3d_kpts['right_shoulder'], :])
            left_shoulders.append(data[start:end, video_pose_3d_kpts['left_shoulder'], :])
            right_hips.append(data[start:end, video_pose_3d_kpts['right_hip'], :])
            left_hips.append(data[start:end, video_pose_3d_kpts['left_hip'], :])
            right_knees.append(data[start:end, video_pose_3d_kpts['right_knee'], :])
            left_knees.append(data[start:end, video_pose_3d_kpts['left_knee'], :])
            right_ankles.append(data[start:end, video_pose_3d_kpts['right_ankle'], :])
            left_ankles.append(data[start:end, video_pose_3d_kpts['left_ankle'], :])

            labels_arr.append(label)
            if len(labels_arr) >= batch_size:
                break

    left_arms = np.concatenate((left_wrists, left_elbows, left_shoulders), axis=2)
    right_arms = np.concatenate((right_wrists, right_elbows, right_shoulders), axis=2)
    left_legs = np.concatenate((left_hips, left_knees, left_ankles), axis=2)
    right_legs = np.concatenate((right_hips, right_knees, right_ankles), axis=2)

    return [left_arms, right_arms, left_legs, right_legs], np.array(labels_arr)
