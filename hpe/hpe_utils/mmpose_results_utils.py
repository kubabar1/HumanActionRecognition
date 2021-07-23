import os
import csv
import numpy as np

keypoints = {
    'mmpose': {
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16
    }
}


def sum_acc_all(pose):
    """Get sum of all accuracy values for given pose - sum of all keypoints accuracy values

    :param pose: pose keypoints - list counting 51 elements (x,y,acc)*17
    :return: sum of all accuracy values
    """
    return sum(pose[2::3])


def order_pose_bbox_for_frame(bbox_list, frame_id):
    """Get list of bboxes ordered by acc for given frame

    :param bbox_list: list of bboxes to order
    :param frame_id: frame id
    :return: list of bboxes ordered by acc or empty list (if bboxes array for given frame is empty)
    """
    poses_array = bbox_list[frame_id]
    poses_array_acc = [i[4] for i in poses_array]
    if len(poses_array):
        _, ordered_poses_list = zip(*sorted(zip(poses_array_acc, poses_array)))
        return list(reversed(ordered_poses_list))
    else:
        return []


def order_pose_bbox_list(bbox_list):
    """Get list of bboxes ordered by acc for all frames

    :param bbox_list: list of bboxes to order
    :return: list of bboxes ordered by acc
    """
    return [order_pose_bbox_for_frame(bbox_list, frame_id) for frame_id in range(len(bbox_list))]


def order_pose_keypoints_for_frame(poses_list, frame_id):
    """Get list of poses ordered by acc for all frames

    :param poses_list: list of poses to order
    :param frame_id: frame id
    :return: list of poses ordered by acc or empty list (if poses array for given frame is empty)
    """
    poses_array = poses_list[frame_id]
    poses_array_acc = [sum_acc_all(i) for i in poses_array]
    if len(poses_array):
        _, ordered_poses_list = zip(*sorted(zip(poses_array_acc, poses_array)))
        return list(reversed(ordered_poses_list))
    else:
        return []


def order_pose_keypoints_list(poses_list):
    """Get list of bboxes ordered by acc for all frames

    :param poses_list: list of bboxes to order
    :return: list of bboxes ordered by acc
    """
    return [order_pose_keypoints_for_frame(poses_list, frame_id) for frame_id in range(len(poses_list))]


def get_best_match(poses_list, frame_id):
    """Get pose with the highest accuracy keypoints sum for given frame

    :param poses_list: list of poses to order
    :param frame_id: frame id
    :return: pose with the highest accuracy keypoints sum
    """
    poses_array = poses_list[frame_id]
    poses_array_acc = [sum_acc_all(i) for i in poses_array]
    if len(poses_array):
        return poses_array[poses_array_acc.index(max(poses_array_acc))]
    else:
        return []


def get_best_match_array(poses_list):
    """Return list of keypoints predicted for each frame with biggest accuracies

    :param poses_list: list of poses to order
    :return: list of keypoints predicted for each frame with biggest accuracies
    """
    return [get_best_match(poses_list, frame_id) for frame_id in range(len(poses_list))]


def read_poses_to_array(poses_directory):
    """Read predicted poses to array of size (poses_count x predicted_frames_count x keypoints_count)

    :param poses_directory: path to directory with predicted poses csv-s
    :return: predicted_kpts where
        predicted_kpts - length is count of detected poses
        predicted_kpts[0] - length is equal to frames count
        predicted_kpts[0][0] - length == 51 (detected keypoints of pose) (17*(x, y, acc))
        predicted_kpts[0][0][0:] - (x, y, acc)
    """
    data = []
    for data_file in sorted(os.listdir(poses_directory)):
        data_path = os.path.join(poses_directory, data_file)
        with open(data_path, 'r') as file:
            tmp = {}
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                tmp[int(row[0].replace('"', ''))] = np.array([q.replace('"', '') for q in row[1:]], dtype='float')
            data.append(tmp)
    return data


def read_bbox_to_array(poses_directory):
    data = []
    for data_file in sorted(os.listdir(poses_directory)):
        data_path = os.path.join(poses_directory, data_file)
        with open(data_path, 'r') as file:
            tmp = {}
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                tmp[int(row[0].replace('"', ''))] = np.array([q.replace('"', '') for q in row[1:]], dtype='float')
            data.append(tmp)
    return data


def get_keypoints_for_all_frames_kpts(poses_directory, kpts_count):
    """Read predicted dataset_utils to array of size (frames_count x poses_count x keypoints_count)

    :param poses_directory: path to directory with predicted poses csv-s
    :param kpts_count: path to keypoints dataset_utils
    :return: predicted_kpts where
        predicted_kpts - length is equal to frames count
        predicted_kpts[0] - length is count of detected poses
        predicted_kpts[0][0] - length == 51 (detected keypoints of pose) (17*(x, y, acc))
        predicted_kpts[0][0][0] - x coordinate of first keypoint
    """
    poses = read_poses_to_array(poses_directory)
    kpts = []
    for i in range(kpts_count):
        tmp = []
        for p in poses:
            if i in p:
                tmp.append(p[i])
        kpts.append(np.array(tmp, dtype=object))
    return kpts


def get_bbox_for_all_frames_kpts(poses_directory, kpts_count):
    poses = read_bbox_to_array(poses_directory)
    kpts = []
    for i in range(kpts_count):
        tmp = []
        for p in poses:
            if i in p:
                tmp.append(p[i])
        kpts.append(np.array(tmp, dtype=object))
    return kpts


def get_keypoints_for_all_frames(poses_directory):
    max_frame_cnt = 0
    for i in [f for f in os.listdir(poses_directory) if os.path.isfile(os.path.join(poses_directory, f))]:
        fc = int(get_last_frame_id(os.path.join(poses_directory, i)))
        if fc > max_frame_cnt:
            max_frame_cnt = fc
    return get_keypoints_for_all_frames_kpts(poses_directory, max_frame_cnt)


def get_bbox_for_all_frames(poses_directory):
    max_frame_cnt = 0
    for i in [f for f in os.listdir(poses_directory) if os.path.isfile(os.path.join(poses_directory, f))]:
        fc = int(get_last_frame_id(os.path.join(poses_directory, i)))
        if fc > max_frame_cnt:
            max_frame_cnt = fc
    return get_bbox_for_all_frames_kpts(poses_directory, max_frame_cnt)


def get_last_frame_id(csv_path):
    csv_reader = csv.reader(open(csv_path, 'r'), delimiter=',')
    last_row = None
    for last_row in csv_reader:
        pass
    return last_row[0] if last_row else None
