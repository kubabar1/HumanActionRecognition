import csv
import os
import time
from random import randrange

import cv2
import numpy as np
from har.impl.hierarchical_rnn.evaluate import fit, load_model
from har.utils.dataset_util import get_berkeley_dataset, SetType, berkeley_mhad_classes, video_pose_3d_kpts, DatasetInputType, \
    exercises_v2_classes, mmpose_kpts


def load_data_berkeley():
    data_paths = []
    for root, dirs, files in os.walk('datasets_processed/berkeley/3D'):
        if not dirs and '/S09/' in root and '/R02' in root and '/Cluster01/' in root:
            data_path = os.path.join(root, '3d_coordinates.npy')
            data_paths.append(data_path)

    data_paths = sorted(data_paths)
    labels = [int(i.split(os.path.sep)[-3][1:]) - 1 for i in data_paths]
    data = load_data_berkeley()
    data_labels = [[labels[it] for _ in range(i.shape[0])] for it, i in enumerate(data)]
    data = np.concatenate(data, axis=0)
    data_labels = np.concatenate(data_labels)
    return data, data_labels


def load_data_exercises():
    data = []
    t = '/home/kuba/workspace/human_action_recognition/HumanActionRecognition/datasets_processed/exercises_v2/3D/A{}/R6/3d_coordinates.npy'
    for i in range(1, 13):
        data.extend(np.load(t.format(str(i).zfill(2))))

    return data


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

def main():
    # model_path = '/home/kuba/workspace/human_action_recognition/research/exercises_v2/STEP/model_hierarchical_rnn_en_10000_bs_128_lr_0.0001_op_RMSPROP_hs_128_it_STEP_momentum_0.9_wd_0_split_20_steps_32_rotations_3D_normalized_1636690685904.pth'
    model_path = '/home/kuba/workspace/human_action_recognition/research/exercises_v2/STEP_60/model_hierarchical_rnn_en_10000_bs_128_lr_0.0001_op_RMSPROP_hs_128_it_STEP_momentum_0.9_wd_0_split_20_steps_60_rotations_3D_normalized_1636825919778.pth'
    # model_path = '/home/kuba/workspace/human_action_recognition/research/exercises_v2/SPLIT_PART_SEQUENCE/model_hierarchical_rnn_en_10000_bs_128_lr_0.0001_op_RMSPROP_hs_128_it_SPLIT_momentum_0.9_wd_0_split_20_steps_60_rotations_3D_normalized_1636840165681.pth'
    model = load_model(model_path, len(exercises_v2_classes))
    # data, data_labels = load_data_berkeley()
    data = np.load(
        '/home/kuba/workspace/human_action_recognition/HumanActionRecognition/datasets_processed/exercises_v2_multiple_actions/3D/multiple_actions_exercises/3d_coordinates.npy')
    # data = load_data_exercises()

    true_video_path = '/home/kuba/workspace/human_action_recognition/datasets/custom/exercises_v2/multiple_actions_exercises.mp4'
    # true_video_path = '/home/kuba/workspace/human_action_recognition/datasets/custom/exercises_v2/multiple_exercises_2.mp4'

    separator = 60
    split = 20

    all_cases = 0
    correct = 0

    cap = cv2.VideoCapture(true_video_path)
    stop = False
    skip = False
    frame_id = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    lineType = 2
    predicted = ''

    pose = read_keypoints_csv_no_keys('/home/kuba/workspace/human_action_recognition/HumanActionRecognition/datasets_processed/exercises_v2_multiple_actions/2D/multiple_actions_exercises/x.csv')

    pose = [np.array(i)[list(mmpose_kpts.values())] for i in pose]

    while True:
        if not frame_id % separator and frame_id > 0:
            test_sequence = data[frame_id - separator:frame_id]
            predicted = fit(exercises_v2_classes, test_sequence, model, video_pose_3d_kpts, input_type=DatasetInputType.STEP, split=split,
                            steps=separator)
        if not stop:
            ret, frame = cap.read()
            if ret:
                kpts = pose[frame_id]
                for k in kpts:
                    cv2.circle(frame, (int(k[0]), int(k[1])), 1, (0, 0, 255), 5)
                cv2.putText(frame,
                            'Frame: {}'.format(frame_id),
                            (480, 70),
                            font,
                            fontScale,
                            (0, 255, 0),
                            lineType)
                cv2.putText(frame,
                            'Action: {}'.format(predicted),
                            (480, 90),
                            font,
                            fontScale,
                            (0, 0, 255),
                            lineType)
                cv2.imshow(os.path.basename(true_video_path), frame)
                frame_id += 1
            else:
                break
            # time.sleep(pause_l)
        ky = cv2.waitKey(25) & 0xFF
        if skip:
            skip = False
            stop = not stop
        if ky == ord('q'):
            break
        if ky == ord(' '):
            stop = not stop
        if ky == 9:
            stop = not stop
            skip = True
    cap.release()
    cv2.destroyAllWindows()

    # for it, frame in enumerate(data):
    #     if not it % separator and it > 0:
    #         test_sequence = data[it - separator:it]
    #         predicted = fit(exercises_v2_classes, test_sequence, model, video_pose_3d_kpts, input_type=DatasetInputType.STEP,
    #                         split=split)
    #         print(predicted)
    #         # all_cases += 1
    #         # tmp = [predicted == berkeley_mhad_classes[k] for k in data_labels[it - separator:it]]
    #         # if sum(tmp) > len(tmp) / 2:
    #         #     correct += 1
    #     # if it >= separator:
    #     #     test_sequence = data[it - separator:it]
    #     #     predicted = fit(berkeley_mhad_classes, test_sequence, model, video_pose_3d_kpts, input_type=DatasetInputType.STEP,
    #     #                     split=split)
    #     #     print(predicted)
    #     #     all_cases += 1
    #     #
    #     #     # if predicted == berkeley_mhad_classes[data_labels[it]]:
    #     #     #     correct += 1
    # # print("{}".format(correct / all_cases))

    # test_data, test_labels = get_berkeley_dataset('datasets_processed/berkeley/3D', set_type=SetType.TEST)
    # random_id = randrange(len(test_labels))
    # test_sequence, test_label = test_data[random_id], test_labels[random_id]
    # model_path = '/home/kuba/workspace/human_action_recognition/research/1_lstm_simple/berkeley_mhad_10k/model_lstm_simple_en_10000_bs_128_lr_0.0001_op_RMSPROP_geo_JOINT_COORDINATE_hs_128_hl_3_it_SPLIT_dropout_0.5_momentum_0.9_wd_0_split_20_steps_32_3D_normalized.pth'
    #
    # print(test_sequence.shape)
    #
    # lstm_simple_model = load_model(model_path, len(berkeley_mhad_classes))
    #
    # predicted = fit(berkeley_mhad_classes, test_sequence, lstm_simple_model, video_pose_3d_kpts)
    #
    # print('CORRECT: {}'.format(berkeley_mhad_classes[test_label]))
    # print('PREDICTED: {}'.format(predicted))


if __name__ == '__main__':
    main()
