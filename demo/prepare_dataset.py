import argparse
import os
import pathlib
import shutil

import cv2
import numpy as np
import tqdm

from pose_2d.hpe_2d_api import load_models, estimate_pose
from pose_3d.hpe_3d_api import process_2d_to_3d, load_model


def get_best_pose_for_frame(poses):
    poses_acc = [np.sum(pose[2::3]) for pose in poses]
    return poses[np.argmax(poses_acc)]


def estimate_pose_2d(pose_model, det_model, input_dataset_path, results_pose_2d):
    if os.path.exists(results_pose_2d):
        shutil.rmtree(results_pose_2d)
        os.mkdir(results_pose_2d)

    print('Estimating poses 2D:')
    progress_bar = tqdm.tqdm(total=sum([len(files) for root, dirs, files in os.walk(input_dataset_path) if not dirs]))

    for root, dirs, files in os.walk(input_dataset_path):
        if not dirs:
            for file in files:
                output_path = os.path.join(results_pose_2d, os.path.relpath(root, input_dataset_path), pathlib.Path(file).stem)
                if not os.path.exists(output_path):
                    pathlib.Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
                input_path = os.path.join(root, file)
                cap = cv2.VideoCapture(input_path)
                all_kpts = []
                while cap.isOpened():
                    flag, frame = cap.read()
                    if not flag:
                        break
                    pose_results = estimate_pose(frame, pose_model, det_model)
                    kpts = get_best_pose_for_frame([pose['keypoints'] for pose in pose_results])
                    all_kpts.append(kpts[:, :2])
                np.save(output_path, all_kpts)
                progress_bar.update(1)
    progress_bar.close()


def estimate_pose_3d(input_pose_2d, results_pose_3d, video_pose_3d_path, model_pos, joints_left, joints_right, frame_width, frame_height):
    if os.path.exists(results_pose_3d):
        shutil.rmtree(results_pose_3d)
        os.mkdir(results_pose_3d)

    print('Estimating poses 3D:')
    progress_bar = tqdm.tqdm(total=sum([len(files) for root, dirs, files in os.walk(input_pose_2d) if not dirs]))

    for root, dirs, files in os.walk(input_pose_2d):
        if not dirs:
            for file in files:
                output_path = os.path.join(results_pose_3d, os.path.relpath(root, input_pose_2d), pathlib.Path(file).stem)
                if not os.path.exists(output_path):
                    pathlib.Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
                input_path = os.path.join(root, file)
                keypoints = np.load(input_path)
                prediction = process_2d_to_3d(video_pose_3d_path, keypoints, model_pos, joints_left, joints_right, frame_width,
                                              frame_height)
                np.save(output_path, prediction)
                progress_bar.update(1)
    progress_bar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mmpose-path', help='Absolute path to VideoPose3D', required=True)
    parser.add_argument('--video-pose-3d-path', help='Absolute path to MMPose', required=True)
    parser.add_argument('--frame-width', help='Width of frame', required=True)
    parser.add_argument('--frame-height', help='Height of frame', required=True)
    parser.add_argument('--input-directory', help='Path to input data', required=True)
    parser.add_argument('--output-directory', help='Path to generated results output directory', required=False, type=str,
                        default='results')
    parser.add_argument('--joints-count', help='Count of joints given as input', required=False, type=int, default=17)
    parser.add_argument('--joints-left', help='Joints from left part of silhouette', required=False, type=int, nargs='+',
                        default=[1, 3, 5, 7, 9, 11, 13, 15])
    parser.add_argument('--joints-right', help='Joints from right part of silhouette', required=False, type=int, nargs='+',
                        default=[2, 4, 6, 8, 10, 12, 14, 16])

    args = parser.parse_args()
    mmpose_path = args.mmpose_path
    video_pose_3d_path = args.video_pose_3d_path
    input_dataset_path = args.input_directory
    output_directory = args.output_directory
    joints_count = int(args.joints_count)
    joints_left = args.joints_left
    joints_right = args.joints_right
    frame_width = int(args.frame_width)
    frame_height = int(args.frame_height)

    results_pose_2d = os.path.join(output_directory, 'pose_2d')
    results_pose_3d = os.path.join(output_directory, 'pose_3d')

    det_model, pose_model = load_models(mmpose_path)
    estimate_pose_2d(pose_model, det_model, input_dataset_path, results_pose_2d)

    model_pos = load_model(video_pose_3d_path, joints_count)
    estimate_pose_3d(results_pose_2d, results_pose_3d, video_pose_3d_path, model_pos, joints_left, joints_right, frame_width, frame_height)


if __name__ == '__main__':
    main()
