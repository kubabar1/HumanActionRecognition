import csv
import math
import os
import shutil
import textwrap
from enum import Enum, auto
from random import randrange

import numpy as np
import tqdm

from har.impl.lstm_simple.utils.descriptors.jjd import calculate_jjd
from har.impl.lstm_simple.utils.descriptors.jjo import calculate_jjo
from har.impl.lstm_simple.utils.descriptors.jld import calculate_jld
from har.impl.lstm_simple.utils.descriptors.lla import calculate_lla
from har.impl.lstm_simple.utils.descriptors.rp import calculate_rp

utd_mhad_classes = [
    'RIGHT_ARM_SWIPE_TO_THE_LEFT',
    'RIGHT_ARM_SWIPE_TO_THE_RIGHT',
    'RIGHT_HAND_WAVE',
    'TWO_HAND_FRONT_CLAP',
    'RIGHT_ARM_THROW',
    'CROSS_ARMS_IN_THE_CHEST',
    'BASKETBALL_SHOOT',
    'RIGHT_HAND_DRAW_X',
    'RIGHT_HAND_DRAW_CIRCLE_CLOCKWISE',
    'RIGHT_HAND_DRAW_CIRCLE_COUNTER_CLOCKWISE',
    'DRAW_TRIANGLE',
    'BOWLING_RIGHT_HAND',
    'FRONT_BOXING',
    'BASEBALL_SWING_FROM_RIGHT',
    'TENNIS_RIGHT_HAND_FOREHAND_SWING',
    'ARM_CURL_TWO_ARMS',
    'TENNIS_SERVE',
    'TWO_HAND_PUSH',
    'RIGHT_HAND_KNOCK_ON_DOOR',
    'RIGHT_HAND_CATCH_AN_OBJECT',
    'RIGHT_HAND_PICK_UP_AND_THROW',
    'JOGGING_IN_PLACE',
    'WALKING_IN_PLACE',
    'SIT_TO_STAND',
    'STAND_TO_SIT',
    'FORWARD_LUNGE_LEFT_FOOT_FORWARD',
    'SQUAT_TWO_ARMS_STRETCH_OUT'
]

ntu_rgbd_classes = [
    'DRINK_WATER',
    'EAT_MEAL',
    'BRUSH_TEETH',
    'BRUSH_HAIR',
    'DROP',
    'PICK_UP',
    'THROW',
    'SIT_DOWN',
    'STAND_UP',
    'CLAPPING',
    'READING',
    'WRITING',
    'TEAR_UP_PAPER',
    'PUT_ON_JACKET',
    'TAKE_OFF_JACKET',
    'PUT_ON_A_SHOE',
    'TAKE_OFF_A_SHOE',
    'PUT_ON_GLASSES',
    'TAKE_OFF_GLASSES',
    'PUT_ON_A_HAT_CAP',
    'TAKE_OFF_A_HAT_CAP',
    'CHEER_UP',
    'HAND_WAVING',
    'KICKING_SOMETHING',
    'REACH_INTO_POCKET',
    'HOPING',
    'JUMP_UP',
    'PHONE_CALL',
    'PLAY_WITH_PHONE_TABLET',
    'TYPE_ON_A_KEYBOARD',
    'POINT_TO_SOMETHING',
    'TAKING_A_SELFIE',
    'CHECK_TIME_FROM_WATCH',
    'RUB_TWO_HANDS',
    'NOD_HEAD_BOW',
    'SHAKE_HEAD',
    'WIPE_FACE',
    'SALUTE',
    'PUT_PALMS_TOGETHER',
    'CROSS_HANDS_IN_FRONT',
    'SNEEZE_COUGH',
    'STAGGERING',
    'FALLING_DOWN',
    'HEADACHE',
    'CHEST_PAIN',
    'BACK_PAIN',
    'NECK_PAIN',
    'NAUSEA_VOMITING',
    'FAN_SELF',
    # 'PUNCH_SLAP',
    # 'KICKING',
    # 'PUSHING',
    # 'PAT_ON_BACK',
    # 'POINT_FINGER',
    # 'HUGGING',
    # 'GIVING_OBJECT',
    # 'TOUCH_POCKET',
    # 'SHAKING_HANDS',
    # 'WALKING_TOWARDS',
    # 'WALKING_APART',
]

berkeley_mhad_classes = [
    'JUMPING_IN_PLACE',
    'JUMPING_JACKS',
    'BENDING_HANDS_UP_ALL_THE_WAY_DOWN',
    'PUNCHING_BOXING',
    'WAVING_TWO_HANDS',
    'WAVING_ONE_HAND_RIGHT',
    'CLAPPING_HANDS',
    'THROWING_A_BALL',
    'SIT_DOWN_THEN_STAND_UP',
    'SIT_DOWN',
    'STAND_UP'
]

video_pose_3d_kpts = {
    'right_wrist': 16,
    'left_wrist': 13,
    'right_elbow': 15,
    'left_elbow': 12,
    'right_shoulder': 14,
    'left_shoulder': 11,
    'right_hip': 1,
    'left_hip': 4,
    'right_knee': 2,
    'left_knee': 5,
    'right_ankle': 3,
    'left_ankle': 6
}

mmpose_kpts = {
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

berkeley_frame_width = 640
berkeley_frame_height = 480

utd_mhad_frame_width = 640
utd_mhad_frame_height = 480


class SetType(Enum):
    TRAINING = auto()
    TEST = auto()
    VALIDATION = auto()


class DatasetInputType(Enum):
    STEP = auto()
    SPLIT = auto()
    TREE = auto()


class GeometricFeature(Enum):
    JOINT_COORDINATE = 'JC'
    RELATIVE_POSITION = 'RP'
    JOINT_JOINT_DISTANCE = 'JJD'
    JOINT_JOINT_ORIENTATION = 'JJO'
    JOINT_LINE_DISTANCE = 'JLD'
    LINE_LINE_ANGLE = 'LLA'
    # JOINT_PLANE_DISTANCE = 'JPD'
    # LINE_PLANE_ANGLE = 'LPA'
    # PLANE_PLANE_ANGLE = 'PPA'


def get_utd_mhad_dataset(dataset_path, train_test_val_ratio=(0.7, 0.2, 0.1), set_type=SetType.TRAINING,
                         data_npy_file_name='3d_coordinates.npy'):
    if not round(sum(train_test_val_ratio), 3) == 1.0:
        raise ValueError('Train/Test/Val ratio must sum to 1')
    training_ratio, test_ratio, validation_ratio = train_test_val_ratio

    data_paths = []

    for root, dirs, files in os.walk(dataset_path):
        if not dirs:
            data_path = os.path.join(root, data_npy_file_name)
            data_paths.append(data_path)

    data_paths = sorted(data_paths)

    actions = []

    for a in np.array_split(data_paths, 27):
        actions += [np.concatenate([a[range(it, len(a), 4)] for it in range(4)])]

    tmp = []

    max_arr = np.max([len(i) for i in actions])

    for i in range(max_arr):
        for a in actions:
            if i < len(a):
                tmp.append(a[i])

    dataset_size = len(tmp)
    training_nb = int(training_ratio * dataset_size)
    test_nb = int(test_ratio * dataset_size)
    validation_nb = int(dataset_size - training_nb - test_nb)

    if training_nb <= 0 or test_nb <= 0 or validation_nb <= 0:
        raise ValueError('Train, test and validation set size must be bigger than 0')

    if set_type == SetType.TRAINING:
        data_paths_res = tmp[:training_nb]
    elif set_type == SetType.TEST:
        data_paths_res = tmp[training_nb:training_nb + test_nb]
    elif set_type == SetType.VALIDATION:
        data_paths_res = tmp[training_nb + test_nb:]
    else:
        raise ValueError('Unknown set type')

    data_list = [np.load(i)[:, :, :] for i in data_paths_res]
    label_list = [int(i.split(os.path.sep)[-4][1:]) - 1 for i in data_paths_res]

    return data_list, label_list


def get_csv_data(data_path):
    with open(data_path, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        arr = np.array([np.array(row if len(row) else np.zeros(51), dtype='float') for row in reader])
    return arr.reshape((-1, 17, 3))


def normalise_skeleton(data_3d, left_hip_index, right_hip_index):
    return scale_skeleton(move_hip_to_center(data_3d, left_hip_index, right_hip_index))


def scale_skeleton(data_3d):
    s = 1 / np.max(np.abs(data_3d))
    return data_3d * s


def move_hip_to_center(data_3d, left_hip_index, right_hip_index):
    dd = data_3d[:, left_hip_index, :] + data_3d[:, right_hip_index, :] / 2
    return np.array([[k - dd[i] for k in d] for i, d in enumerate(data_3d)])


def get_berkeley_dataset(dataset_path, train_test_val_ratio=(0.7, 0.2, 0.1), set_type=SetType.TRAINING,
                         data_file_name=None, use_3d=True):
    if data_file_name is None:
        if use_3d:
            data_file_name = '3d_coordinates.npy'
        else:
            data_file_name = 'x.csv'

    if not round(sum(train_test_val_ratio), 3) == 1.0:
        raise ValueError('Train/Test/Val ratio must sum to 1')
    training_ratio, test_ratio, validation_ratio = train_test_val_ratio

    data_paths = []

    for root, dirs, files in os.walk(dataset_path):
        if not dirs:
            data_path = os.path.join(root, data_file_name)
            data_paths.append(data_path)

    data_paths = sorted(data_paths)
    data_paths_res = []

    # cluster1 = list(filter(lambda x: '/Cluster01/' in x, data_paths))
    # cluster2 = list(filter(lambda x: '/Cluster02/' in x, data_paths))
    # cluster3 = list(filter(lambda x: '/Cluster03/' in x, data_paths))
    # cluster4 = list(filter(lambda x: '/Cluster04/' in x, data_paths))

    for i in np.array_split(data_paths, 4):
        dataset_size = len(i)
        training_nb = int(training_ratio * dataset_size)
        test_nb = int(test_ratio * dataset_size)
        validation_nb = int(dataset_size - training_nb - test_nb)

        if training_nb <= 0 or test_nb <= 0 or validation_nb <= 0:
            raise ValueError('Train, test and validation set size must be bigger than 0')

        if set_type == SetType.TRAINING:
            data_paths_res.extend(i[:training_nb])
        elif set_type == SetType.TEST:
            data_paths_res.extend(i[training_nb:training_nb + test_nb])
        elif set_type == SetType.VALIDATION:
            data_paths_res.extend(i[training_nb + test_nb:])
        else:
            raise ValueError('Unknown set type')

    if use_3d:
        data_list = [np.load(i)[:, :, :] for i in data_paths_res]
    else:
        data_list = [normalize_screen_coordinates(get_csv_data(i)[:, :, :2], berkeley_frame_width, berkeley_frame_height) for i in
                     data_paths_res]
    label_list = [int(i.split(os.path.sep)[-3][1:]) - 1 for i in data_paths_res]

    return data_list, label_list


# def get_berkeley_custom_dataset(dataset_path):
#     data_file_name = '3d_coordinates.npy'
#     data_paths = []
#     for root, dirs, files in os.walk(dataset_path):
#         if not dirs:
#             data_path = os.path.join(root, data_file_name)
#             data_paths.append(data_path)
#
#     data_paths = sorted(data_paths)
#     data_list = [np.load(i)[:, :, :] / 1.5 for i in data_paths]
#     label_list = [int(i.split(os.path.sep)[-3][1:]) - 1 for i in data_paths]
#
#     return data_list, label_list


def get_ntu_rgbd_dataset(dataset_path, train_test_val_ratio=(0.7, 0.2, 0.1), set_type=SetType.TRAINING,
                         data_file_name=None, use_3d=True):
    if data_file_name is None:
        if use_3d:
            data_file_name = '3d_coordinates.npy'
        else:
            data_file_name = 'x.csv'
    if not round(sum(train_test_val_ratio), 3) == 1.0:
        raise ValueError('Train/Test/Val ratio must sum to 1')
    training_ratio, test_ratio, validation_ratio = train_test_val_ratio

    data_paths = []

    for root, dirs, files in os.walk(dataset_path):
        if not dirs:
            action_id = int(textwrap.wrap(root.split('/')[-1].split('_')[0], 4)[-1][1:])
            if action_id < 50:
                data_path = os.path.join(root, data_file_name)
                data_paths.append(data_path)

    data_paths = sorted(data_paths)
    data_paths_res = []

    for i in np.array_split(data_paths, 3):
        dataset_size = len(i)
        training_nb = int(training_ratio * dataset_size)
        test_nb = int(test_ratio * dataset_size)
        validation_nb = int(dataset_size - training_nb - test_nb)

        if training_nb <= 0 or test_nb <= 0 or validation_nb <= 0:
            raise ValueError('Train, test and validation set size must be bigger than 0')

        if set_type == SetType.TRAINING:
            data_paths_res.extend(i[:training_nb])
        elif set_type == SetType.TEST:
            data_paths_res.extend(i[training_nb:training_nb + test_nb])
        elif set_type == SetType.VALIDATION:
            data_paths_res.extend(i[training_nb + test_nb:])
        else:
            raise ValueError('Unknown set type')

    if use_3d:
        data_list = [np.load(i)[:, :, :] for i in data_paths_res]
    else:
        data_list = [normalize_screen_coordinates(get_csv_data(i)[:, :, :2], berkeley_frame_width, berkeley_frame_height) for i in
                     data_paths_res]
    label_list = [int(textwrap.wrap(i.split('/')[-2].split('_')[0], 4)[-1][1:]) - 1 for i in data_paths_res]

    return data_list, label_list


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]


def normalise_2d_data(keypoints, video_width, video_height, analysed_kpts_description=video_pose_3d_kpts):
    keypoints = keypoints[:, list(analysed_kpts_description.values()), :2]
    return normalize_screen_coordinates(keypoints[..., :2], w=video_width, h=video_height)


def random_rotate_y(data_3d, rotation=None):
    if rotation is None:
        rotation = [np.radians(i) for i in np.array(range(0, 360, 15))]
    rotation = Ry(rotation[randrange(len(rotation))])
    return np.array([frame * rotation for frame in data_3d])


def Ry(theta):
    return np.matrix([[math.cos(theta), 0, math.sin(theta)],
                      [0, 1, 0],
                      [-math.sin(theta), 0, math.cos(theta)]])


def get_all_body_parts_steps(data, analysed_body_parts, analysed_kpts_description, begin, end):
    body_parts = [body_part_steps(data, analysed_kpts_description[bp], begin, end) for bp in analysed_body_parts]
    return dict(zip(analysed_body_parts, body_parts))


def get_all_body_parts_splits(data, analysed_body_parts, analysed_kpts_description, split):
    body_parts = [body_part_splits(data, analysed_kpts_description[bp], split) for bp in analysed_body_parts]
    return dict(zip(analysed_body_parts, body_parts))


def body_part_steps(data, analysed_kpt_id, begin, end):
    return data[begin:end, analysed_kpt_id, :]


def body_part_splits(data, analysed_kpt_id, split):
    return np.array([a[randrange(len(a))] for a in np.array_split(data[:, analysed_kpt_id, :], split)])


def prepare_body_part_data(data, analysed_body_parts, analysed_kpts_description, input_type, steps, split):
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

    if input_type == DatasetInputType.STEP:
        parts = int(data.shape[0] / steps)
        for i in range(parts):
            begin = i * steps
            end = i * steps + steps
            body_el = get_all_body_parts_steps(data, analysed_body_parts, analysed_kpts_description, begin, end)
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
    elif input_type == DatasetInputType.SPLIT:
        body_el = get_all_body_parts_splits(data, analysed_body_parts, analysed_kpts_description, split)
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

    left_arms = np.concatenate((left_wrists, left_elbows, left_shoulders), axis=2)
    right_arms = np.concatenate((right_wrists, right_elbows, right_shoulders), axis=2)
    left_legs = np.concatenate((left_hips, left_knees, left_ankles), axis=2)
    right_legs = np.concatenate((right_hips, right_knees, right_ankles), axis=2)

    return [left_arms, right_arms, left_legs, right_legs]


def get_left_kpts(analysed_kpts_description):
    return [
        analysed_kpts_description['left_wrist'],
        analysed_kpts_description['left_elbow'],
        analysed_kpts_description['left_shoulder'],
        analysed_kpts_description['left_hip'],
        analysed_kpts_description['left_knee'],
        analysed_kpts_description['left_ankle']
    ]


def get_right_kpts(analysed_kpts_description):
    return [
        analysed_kpts_description['right_wrist'],
        analysed_kpts_description['right_elbow'],
        analysed_kpts_description['right_shoulder'],
        analysed_kpts_description['right_hip'],
        analysed_kpts_description['right_knee'],
        analysed_kpts_description['right_ankle']
    ]


def prepare_dataset(data, labels, set_type, analysed_kpts_description, geometric_feature, use_cache, remove_cache, method_name):
    dataset_cache_root_dir = 'dataset_cache'
    dataset_cache_method_dir = method_name
    dataset_cache_data_file_name = 'data_arr_cache'
    dataset_cache_labels_file_name = 'labels_arr_cache'
    dataset_cache_dir = os.path.join(dataset_cache_root_dir, dataset_cache_method_dir, geometric_feature.value, set_type.name)
    data_arr_cache_path = os.path.join(dataset_cache_dir, dataset_cache_data_file_name)
    labels_arr_cache_path = os.path.join(dataset_cache_dir, dataset_cache_labels_file_name)

    if remove_cache:
        shutil.rmtree(dataset_cache_dir)

    if use_cache and os.path.exists(data_arr_cache_path + '.npy') and os.path.exists(labels_arr_cache_path + '.npy'):
        data_arr = np.load(data_arr_cache_path + '.npy', allow_pickle=True)
        labels_arr = np.load(labels_arr_cache_path + '.npy', allow_pickle=True)
    else:
        print('Generating dataset ...')
        progress_bar = tqdm.tqdm(total=len(data))

        data_arr = []
        labels_arr = []

        for it in range(len(data)):
            data_arr.append(get_data_for_geometric_type(data[it], analysed_kpts_description, geometric_feature))
            labels_arr.append(labels[it])
            progress_bar.update(1)
        progress_bar.close()

        if use_cache:
            if not os.path.exists(dataset_cache_dir):
                os.makedirs(dataset_cache_dir)
            np.save(data_arr_cache_path, np.array(data_arr, dtype=object))
            np.save(labels_arr_cache_path, np.array(labels_arr, dtype=object))

        print('Dataset generated')

    return data_arr, labels_arr


def get_data_for_geometric_type(data_el, analysed_kpts_description, geometric_feature):
    all_analysed_kpts = list(analysed_kpts_description.values())
    if geometric_feature == GeometricFeature.RELATIVE_POSITION:
        return calculate_rp(data_el, all_analysed_kpts)
    elif geometric_feature == GeometricFeature.JOINT_JOINT_DISTANCE:
        return calculate_jjd(data_el, all_analysed_kpts)
    elif geometric_feature == GeometricFeature.JOINT_JOINT_ORIENTATION:
        return calculate_jjo(data_el, all_analysed_kpts)
    elif geometric_feature == GeometricFeature.JOINT_LINE_DISTANCE:
        return calculate_jld(data_el, all_analysed_kpts, get_analysed_lines_ids(analysed_kpts_description))
    elif geometric_feature == GeometricFeature.LINE_LINE_ANGLE:
        return calculate_lla(data_el, get_analysed_lines_ids(analysed_kpts_description))
    else:
        raise ValueError('Invalid or unimplemented geometric feature type')


def get_analysed_lines_ids(analysed_kpts_description):
    analysed_lines_1 = [
        [analysed_kpts_description['right_wrist'], analysed_kpts_description['right_elbow']],
        [analysed_kpts_description['right_elbow'], analysed_kpts_description['right_shoulder']],
        [analysed_kpts_description['left_wrist'], analysed_kpts_description['left_elbow']],
        [analysed_kpts_description['left_elbow'], analysed_kpts_description['left_shoulder']],
        [analysed_kpts_description['right_hip'], analysed_kpts_description['right_knee']],
        [analysed_kpts_description['right_knee'], analysed_kpts_description['right_ankle']],
        [analysed_kpts_description['left_hip'], analysed_kpts_description['left_knee']],
        [analysed_kpts_description['left_knee'], analysed_kpts_description['left_ankle']]
    ]
    analysed_lines_2 = [
        [analysed_kpts_description['right_wrist'], analysed_kpts_description['right_shoulder']],
        [analysed_kpts_description['left_wrist'], analysed_kpts_description['left_shoulder']],
        [analysed_kpts_description['right_hip'], analysed_kpts_description['right_ankle']],
        [analysed_kpts_description['left_hip'], analysed_kpts_description['left_ankle']]
    ]
    analysed_lines_3 = [
        [analysed_kpts_description['right_wrist'], analysed_kpts_description['left_wrist']],
        [analysed_kpts_description['right_ankle'], analysed_kpts_description['left_ankle']],
        [analysed_kpts_description['right_wrist'], analysed_kpts_description['right_ankle']],
        [analysed_kpts_description['right_wrist'], analysed_kpts_description['left_ankle']],
        [analysed_kpts_description['left_wrist'], analysed_kpts_description['right_ankle']],
        [analysed_kpts_description['left_wrist'], analysed_kpts_description['left_ankle']]
    ]
    return analysed_lines_1 + analysed_lines_2 + analysed_lines_3
