import argparse
import time
from configparser import ConfigParser

import cv2
import numpy as np

from har.utils.dataset_util import exercises_v2_classes, video_pose_3d_kpts, DatasetInputType
from pose_2d.hpe_2d_api import load_models, estimate_pose, get_best_pose_for_frame, get_best_bbox_for_frame
from pose_3d.hpe_3d_api import load_model, process_2d_to_3d
import har.impl.hierarchical_rnn.evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mmpose-path', help='Absolute path to VideoPose3D', required=False)
    parser.add_argument('--video-pose-3d-path', help='Absolute path to MMPose', required=False)
    parser.add_argument('--joints-count', help='Count of joints given as input', required=False, type=int, default=17)
    parser.add_argument('--joints-left', help='Joints from left part of silhouette', required=False, type=int, nargs='+',
                        default=[1, 3, 5, 7, 9, 11, 13, 15])
    parser.add_argument('--joints-right', help='Joints from right part of silhouette', required=False, type=int, nargs='+',
                        default=[2, 4, 6, 8, 10, 12, 14, 16])

    args = parser.parse_args()
    mmpose_path = args.mmpose_path
    video_pose_3d_path = args.video_pose_3d_path

    joints_count = int(args.joints_count)
    joints_left = args.joints_left
    joints_right = args.joints_right

    cap = cv2.VideoCapture(0)

    if mmpose_path is None:
        config = ConfigParser()
        config.read('../config.ini')
        mmpose_path = config.get('main', 'MMPOSE_PATH')

    if video_pose_3d_path is None:
        config = ConfigParser()
        config.read('../config.ini')
        video_pose_3d_path = config.get('main', 'VIDEO_POSE_3D_PATH')

    det_model, pose_model = load_models(mmpose_path, hpe_method='mobilenetv2_coco_384x288')
    video_pose_model = load_model(video_pose_3d_path, joints_count)

    har_model_path = '/home/kuba/workspace/human_action_recognition/research/exercises_v2/STEP_60/model_hierarchical_rnn_en_10000_bs_128_lr_0.0001_op_RMSPROP_hs_128_it_STEP_momentum_0.9_wd_0_split_20_steps_60_rotations_3D_normalized_1636825919778.pth'
    har_model = har.impl.hierarchical_rnn.evaluate.load_model(har_model_path, len(exercises_v2_classes))

    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    prev_frame_time = 0

    separator = 60
    split = 20

    data = []

    predicted = ''
    accuracy = 0

    frame_id = 0
    while True:
        ret, frame = cap.read()

        pose_results = estimate_pose(frame, pose_model, det_model)
        if len(pose_results):
            kpts = get_best_pose_for_frame([pose['keypoints'] for pose in pose_results])[:, :2]
            bbox = get_best_bbox_for_frame([pose['bbox'] for pose in pose_results])[:4]
            for k in kpts:
                cv2.circle(frame, (int(k[0]), int(k[1])), 1, (0, 0, 255), 5)
            if len(bbox):
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.line(frame, (x1, y1), (x2, y1), (0, 255, 0), thickness=1)
                cv2.line(frame, (x1, y2), (x2, y2), (0, 255, 0), thickness=1)
                cv2.line(frame, (x1, y1), (x1, y2), (0, 255, 0), thickness=1)
                cv2.line(frame, (x2, y1), (x2, y2), (0, 255, 0), thickness=1)

            kpts_3d = process_2d_to_3d(video_pose_3d_path, np.array([kpts]), video_pose_model, joints_left, joints_right, frame_width,
                                       frame_height)

            print(kpts_3d.shape)
            
            if not frame_id % separator and frame_id > 0:
                data = np.concatenate(np.array(data))
                predicted, accuracy = har.impl.hierarchical_rnn.evaluate.fit(exercises_v2_classes, data, har_model, video_pose_3d_kpts,
                                                                             input_type=DatasetInputType.STEP, split=split, steps=separator)
                data = []
            data.append(kpts_3d)
        frame_id += 1

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        cv2.putText(frame,
                    'FPS: {}'.format(str(round(fps, 2))),
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2)

        cv2.putText(frame,
                    'Action: {}'.format(predicted),
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2)

        cv2.putText(frame,
                    'Accuracy: {}'.format(str(round(accuracy, 2))),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
