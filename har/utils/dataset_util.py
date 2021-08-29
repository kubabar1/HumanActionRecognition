import os
import textwrap
from enum import Enum, auto

import numpy as np

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


class SetType(Enum):
    TRAINING = auto()
    TEST = auto()
    VALIDATION = auto()


class DatasetInputType(Enum):
    STEP = auto()
    SPLIT = auto()


def get_analysed_keypoints(is_3d=True):
    keypoints = video_pose_3d_kpts if is_3d else mmpose_kpts
    analysed_kpts_left = [
        keypoints['left_hip'], keypoints['left_knee'], keypoints['left_ankle'],
        keypoints['left_shoulder'], keypoints['left_elbow'], keypoints['left_wrist']
    ]
    analysed_kpts_right = [
        keypoints['right_hip'], keypoints['right_knee'], keypoints['right_ankle'],
        keypoints['right_shoulder'], keypoints['right_elbow'], keypoints['right_wrist']
    ]
    return analysed_kpts_left, analysed_kpts_right


def get_utd_mhad_dataset_3d(dataset_path, train_test_val_ratio=(0.7, 0.2, 0.1), set_type=SetType.TRAINING,
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

    data_paths_res = []

    for i in np.array_split(data_paths, 11):
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

    data_list = [np.load(i)[:, :, :] for i in data_paths_res]
    label_list = [int(i.split(os.path.sep)[-4][1:]) - 1 for i in data_paths_res]

    return data_list, label_list


def get_berkeley_dataset_3d(dataset_path, train_test_val_ratio=(0.7, 0.2, 0.1), set_type=SetType.TRAINING,
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
    data_paths_res = []

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

    data_list = [np.load(i)[:, :, :] for i in data_paths_res]
    label_list = [int(i.split(os.path.sep)[-3][1:]) - 1 for i in data_paths_res]

    return data_list, label_list


def get_ntu_rgbd_dataset_3d(dataset_path, train_test_val_ratio=(0.7, 0.2, 0.1), set_type=SetType.TRAINING,
                            data_npy_file_name='3d_coordinates.npy'):
    if not round(sum(train_test_val_ratio), 3) == 1.0:
        raise ValueError('Train/Test/Val ratio must sum to 1')
    training_ratio, test_ratio, validation_ratio = train_test_val_ratio

    data_paths = []

    for root, dirs, files in os.walk(dataset_path):
        if not dirs:
            action_id = int(textwrap.wrap(root.split('/')[-1].split('_')[0], 4)[-1][1:])
            if action_id < 50:
                data_path = os.path.join(root, data_npy_file_name)
                data_paths.append(data_path)

    dataset_size = len(data_paths)
    training_nb = int(training_ratio * dataset_size)
    test_nb = int(test_ratio * dataset_size)
    validation_nb = int(dataset_size - training_nb - test_nb)

    if training_nb <= 0 or test_nb <= 0 or validation_nb <= 0:
        raise ValueError('Train, test and validation set size must be bigger than 0')

    data_paths = sorted(data_paths)

    if set_type == SetType.TRAINING:
        data_paths = data_paths[:training_nb]
    elif set_type == SetType.TEST:
        data_paths = data_paths[training_nb:training_nb + test_nb]
    elif set_type == SetType.VALIDATION:
        data_paths = data_paths[training_nb + test_nb:]
    else:
        raise ValueError('Unknown set type')

    data_list = [np.load(i)[:, :, :] for i in data_paths]
    label_list = [int(textwrap.wrap(i.split('/')[-2].split('_')[0], 4)[-1][1:]) - 1 for i in data_paths]

    return data_list, label_list


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]


def normalise_2d_data(keypoints, video_width, video_height, analysed_kpts_description=video_pose_3d_kpts):
    keypoints = keypoints[:, list(analysed_kpts_description.values()), :2]
    return normalize_screen_coordinates(keypoints[..., :2], w=video_width, h=video_height)
