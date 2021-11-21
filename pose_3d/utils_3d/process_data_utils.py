import csv
import os
import pathlib
import shutil

import numpy as np
import tqdm

from har_pose_3d.run_video_pose_3d import process_2d_to_3d, load_model
from .mmpose_results_utils import get_keypoints_for_all_frames


def get_best_pose_for_frame(poses):
    tmp = []
    best_acc = 0
    for pose in poses:
        acc = np.sum(pose[2::3])
        if acc > best_acc:
            best_acc = acc
            tmp = pose
    return tmp


def process_to_2d_single_person(input_data_path, output_directory):
    output_directory = os.path.join(output_directory, '2D')
    output_file_name = 'x.csv'

    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)

    for root, dirs, files in os.walk(input_data_path):
        if not dirs:
            input_file = root
            input_rel_path = os.path.relpath(root, input_data_path)
            output_path = os.path.join(output_directory, input_rel_path, output_file_name)

            if not os.path.exists(os.path.dirname(output_path)):
                pathlib.Path(os.path.dirname(output_path)).mkdir(parents=True)
            final_pose = [get_best_pose_for_frame(poses) if len(poses) else [] for poses in
                          get_keypoints_for_all_frames(input_file)]
            with open(output_path, "w+") as my_csv:
                csvWriter = csv.writer(my_csv, delimiter=',')
                csvWriter.writerows(final_pose)


def process_to_3d_single_person(video_pose_3d_path, input_data_path, output_directory, frame_width, frame_height, joints_count, joints_left,
                                joints_right):
    output_directory = os.path.join(output_directory, '3D')
    output_file_name = '3d_coordinates'
    model_pos = load_model(video_pose_3d_path, joints_count)

    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)

    count = sum([1 for root, dirs, files in os.walk(input_data_path) if not dirs])
    progress_bar = tqdm.tqdm(total=count)

    for root, dirs, files in os.walk(input_data_path):
        if not dirs:
            input_file = root
            input_rel_path = os.path.relpath(root, input_data_path)
            output_path = os.path.join(output_directory, input_rel_path, output_file_name)

            if not os.path.exists(output_path):
                pathlib.Path(os.path.dirname(output_path)).mkdir(parents=True)

            keypoints = np.array(
                [
                    np.array([np.array([i[0], i[1]], dtype='float')
                              for i in np.array_split(frame_poses[0], 17)], dtype='float')
                    if len(frame_poses)
                    else np.zeros((17, 2), dtype='float')
                    for frame_poses
                    in get_keypoints_for_all_frames(input_file)], dtype='float')

            prediction = process_2d_to_3d(video_pose_3d_path, keypoints, model_pos, joints_left, joints_right, frame_width, frame_height)

            np.save(output_path, prediction)
            progress_bar.update(1)

    progress_bar.close()
