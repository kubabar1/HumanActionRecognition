import csv
import os

import numpy as np


def read_keypoints_csv(input_csv_path):
    """Read predicted keypoints saved in csv file

    :param input_csv_path: path to input csv with predicted keypoints
    :return: {'frame_nb': [ (keypoint_1_x, keypoint_1_y, keypoint_1_accuracy), (keypoint_2_x, ...), ...], 'frame_nb': [...], ...}
        e.g:
        {'1': [('12.2', '43.2', '0.342312'), ('2.1', '3.4', '0.41232'), ...], '2': [('33.2', '1.5', '0.64312'), ...], ... }
    """
    result = {}
    with open(input_csv_path, newline='') as csvfile:
        keypoints_reader = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in keypoints_reader:
            result[int(float(row[0]))] = [np.array(row[i * 3 + 1:i * 3 + 4], dtype='float') for i in range(17)]
            i += 1
    return result


def read_keypoints_csv_no_keys(input_csv_path):
    """Read predicted keypoints saved in csv file

    :param input_csv_path: path to input csv with predicted keypoints
    :return: {'frame_nb': [ (keypoint_1_x, keypoint_1_y, keypoint_1_accuracy), (keypoint_2_x, ...), ...], 'frame_nb': [...], ...}
        e.g:
        {'1': [('12.2', '43.2', '0.342312'), ('2.1', '3.4', '0.41232'), ...], '2': [('33.2', '1.5', '0.64312'), ...], ... }
    """
    result = []
    with open(input_csv_path, newline='') as csvfile:
        keypoints_reader = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in keypoints_reader:
            result.append([np.array(row[i * 3:i * 3 + 3], dtype='float') for i in range(17)])
            i += 1
    return result


def read_keypoints_all_poses(pose_result_dir_path):
    """Read keypoints for all poses in directory

    :param pose_result_dir_path: directory containing predicted poses
    :return: array of poses with keypoints
    """
    poses_csv = sorted([os.path.join(pose_result_dir_path, f) for f in os.listdir(pose_result_dir_path) if
                        os.path.isfile(os.path.join(pose_result_dir_path, f))])
    poses_keypoints = []
    for pose_file_csv in poses_csv:
        poses_keypoints.append(read_keypoints_csv(pose_file_csv))
    return poses_keypoints


def read_bbox_csv(input_csv_path):
    """Read predicted bboxes saved in csv file

    :param input_csv_path: path to input csv with predicted bbox-es
    :return: {'frame_nb': [x, y, x, y, accuracy], 'frame_nb': [x, y, x, y, accuracy], ....}
    """
    result = {}
    with open(input_csv_path, newline='') as csvfile:
        bbox_reader = csv.reader(csvfile, delimiter=',')
        for row in bbox_reader:
            result[row[0]] = row[1:]
    return result


def read_bboxes_all_poses(pose_result_dir_path):
    """Read predicted bboxes for all poses in given directory

    :param pose_result_dir_path: path to input dir with csv-s containing poses with keypoints
    :return: array containing bbox-es for all poses [pose1, pose2, pose3]
        pose -> {'frame_nb': [x, y, x, y, accuracy], 'frame_nb': [x, y, x, y, accuracy], ....}
    """
    poses_csv = [os.path.join(pose_result_dir_path, f) for f in os.listdir(pose_result_dir_path) if
                 os.path.isfile(os.path.join(pose_result_dir_path, f))]
    poses_bboxes = []
    for pose_file_csv in poses_csv:
        poses_bboxes.append(read_bbox_csv(pose_file_csv))
    return poses_bboxes


def sum_acc_all(pose):
    """Get sum of all accuracy values for given pose - sum of all keypoints accuracy values

    :param pose: pose keypoints - list counting 51 elements (x,y,acc)*17
    :return: sum of all accuracy values
    """
    return np.sum(pose[2::3])


def get_best_pose(poses, frames_count):
    """Get best pose for each frame in sequence

    :param poses: array of detected poses
    :param frames_count: count of frames
    :return: array of keypoints for each sequence frame, for only one person (with best prediction)
        [!IMPORTANT if pose wasn't detected for frame, it'll contain empty array]
    """
    res = []
    for f in range(frames_count):
        tmp = []
        best_acc = 0
        for pose in poses:
            if f in pose:
                acc = np.sum([float(i[2]) for i in pose[f]])
                if acc > best_acc:
                    best_acc = acc
                    tmp = pose[f]
        res.append(tmp)
    return np.array(res)
