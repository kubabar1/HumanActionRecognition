import os
from random import randrange

import numpy as np

from har.utils.dataset_utils import video_pose_3d_kpts


def get_batch_ntu_rgbd(dataset_path, batch_size=128, split=8, is_training=True, data_npy_file_name='3d_coordinates.npy'):
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

        right_wrists.append(np.array(
            [a[randrange(len(a))] for a in np.array_split(data[:, video_pose_3d_kpts['right_wrist'], :], split)]))
        left_wrists.append(np.array(
            [a[randrange(len(a))] for a in np.array_split(data[:, video_pose_3d_kpts['left_wrist'], :], split)]))
        right_elbows.append(np.array(
            [a[randrange(len(a))] for a in np.array_split(data[:, video_pose_3d_kpts['right_elbow'], :], split)]))
        left_elbows.append(np.array(
            [a[randrange(len(a))] for a in np.array_split(data[:, video_pose_3d_kpts['left_elbow'], :], split)]))
        right_shoulders.append(np.array(
            [a[randrange(len(a))] for a in np.array_split(data[:, video_pose_3d_kpts['right_shoulder'], :], split)]))
        left_shoulders.append(np.array(
            [a[randrange(len(a))] for a in np.array_split(data[:, video_pose_3d_kpts['left_shoulder'], :], split)]))
        right_hips.append(
            np.array([a[randrange(len(a))] for a in np.array_split(data[:, video_pose_3d_kpts['right_hip'], :], split)]))
        left_hips.append(
            np.array([a[randrange(len(a))] for a in np.array_split(data[:, video_pose_3d_kpts['left_hip'], :], split)]))
        right_knees.append(np.array(
            [a[randrange(len(a))] for a in np.array_split(data[:, video_pose_3d_kpts['right_knee'], :], split)]))
        left_knees.append(
            np.array([a[randrange(len(a))] for a in np.array_split(data[:, video_pose_3d_kpts['left_knee'], :], split)]))
        right_ankles.append(np.array(
            [a[randrange(len(a))] for a in np.array_split(data[:, video_pose_3d_kpts['right_ankle'], :], split)]))
        left_ankles.append(np.array(
            [a[randrange(len(a))] for a in np.array_split(data[:, video_pose_3d_kpts['left_ankle'], :], split)]))

        labels_arr.append(label)
        if len(labels_arr) >= batch_size:
            break

    left_arms = np.concatenate((left_wrists, left_elbows, left_shoulders), axis=2)
    right_arms = np.concatenate((right_wrists, right_elbows, right_shoulders), axis=2)
    left_legs = np.concatenate((left_hips, left_knees, left_ankles), axis=2)
    right_legs = np.concatenate((right_hips, right_knees, right_ankles), axis=2)

    return [left_arms, right_arms, left_legs, right_legs], np.array(labels_arr)


def get_batch_berkeley_mhad(dataset_path, batch_size=128, split=20, is_training=True, data_npy_file_name='3d_coordinates.npy'):
    clusters_count = 1  # 4
    cameras_count = 1
    persons_count = 10  # 12
    actions_count = 11
    repetitions_count = 5

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
        cluster_id = randrange(clusters_count) + 1
        cam_id = cameras_count
        person_id = randrange(8) + 1 if is_training else randrange(8, persons_count) + 1
        action_id = randrange(actions_count) + 1
        repetition_id = randrange(repetitions_count) + 1

        data_path = os.path.join(dataset_path,
                                 'Cluster{}'.format(str(cluster_id).zfill(2)),
                                 'Cam{}'.format(str(cam_id).zfill(2)),
                                 'S{}'.format(str(person_id).zfill(2)),
                                 'A{}'.format(str(action_id).zfill(2)),
                                 'R{}'.format(str(repetition_id).zfill(2)),
                                 data_npy_file_name)

        if not os.path.exists(data_path):
            continue

        data = np.array(np.load(data_path))
        label = action_id - 1

        right_wrists.append(np.array(
            [a[randrange(len(a))] for a in np.array_split(data[:, video_pose_3d_kpts['right_wrist'], :], split)]))
        left_wrists.append(np.array(
            [a[randrange(len(a))] for a in np.array_split(data[:, video_pose_3d_kpts['left_wrist'], :], split)]))
        right_elbows.append(np.array(
            [a[randrange(len(a))] for a in np.array_split(data[:, video_pose_3d_kpts['right_elbow'], :], split)]))
        left_elbows.append(np.array(
            [a[randrange(len(a))] for a in np.array_split(data[:, video_pose_3d_kpts['left_elbow'], :], split)]))
        right_shoulders.append(np.array(
            [a[randrange(len(a))] for a in np.array_split(data[:, video_pose_3d_kpts['right_shoulder'], :], split)]))
        left_shoulders.append(np.array(
            [a[randrange(len(a))] for a in np.array_split(data[:, video_pose_3d_kpts['left_shoulder'], :], split)]))
        right_hips.append(
            np.array([a[randrange(len(a))] for a in np.array_split(data[:, video_pose_3d_kpts['right_hip'], :], split)]))
        left_hips.append(
            np.array([a[randrange(len(a))] for a in np.array_split(data[:, video_pose_3d_kpts['left_hip'], :], split)]))
        right_knees.append(np.array(
            [a[randrange(len(a))] for a in np.array_split(data[:, video_pose_3d_kpts['right_knee'], :], split)]))
        left_knees.append(
            np.array([a[randrange(len(a))] for a in np.array_split(data[:, video_pose_3d_kpts['left_knee'], :], split)]))
        right_ankles.append(np.array(
            [a[randrange(len(a))] for a in np.array_split(data[:, video_pose_3d_kpts['right_ankle'], :], split)]))
        left_ankles.append(np.array(
            [a[randrange(len(a))] for a in np.array_split(data[:, video_pose_3d_kpts['left_ankle'], :], split)]))

        labels_arr.append(label)
        if len(labels_arr) >= batch_size:
            break

    left_arms = np.concatenate((left_wrists, left_elbows, left_shoulders), axis=2)
    right_arms = np.concatenate((right_wrists, right_elbows, right_shoulders), axis=2)
    left_legs = np.concatenate((left_hips, left_knees, left_ankles), axis=2)
    right_legs = np.concatenate((right_hips, right_knees, right_ankles), axis=2)

    return [left_arms, right_arms, left_legs, right_legs], np.array(labels_arr)
