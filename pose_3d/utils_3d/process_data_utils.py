import csv
import os
import pathlib
import shutil
import sys

import numpy as np

from .mmpose_results_utils import get_keypoints_for_all_frames


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
            final_pose = [poses[0] if len(poses) else [] for poses in get_keypoints_for_all_frames(input_file)]
            with open(output_path, "w+") as my_csv:
                csvWriter = csv.writer(my_csv, delimiter=',')
                csvWriter.writerows(final_pose)


def process_to_3d_single_person(input_data_path, output_directory, frame_width, frame_height, video_pose_3d_path):
    tmp_3d_results_path = os.path.join(output_directory, 'tmp')
    output_directory = os.path.join(output_directory, '3D')
    output_file_name = '3d_coordinates'

    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)

    viz_subject = 'VideoPose3D'
    viz_action = 'custom'

    for root, dirs, files in os.walk(input_data_path):
        if not dirs:
            input_file = root
            input_rel_path = os.path.relpath(root, input_data_path)
            output_path = os.path.join(tmp_3d_results_path, input_rel_path, viz_subject)

            if not os.path.exists(os.path.dirname(output_path)):
                pathlib.Path(os.path.dirname(output_path)).mkdir(parents=True)

            final_poses = np.array(
                [
                    np.array([np.array([i[0], i[1]], dtype='float')
                              for i in np.array_split(frame_poses[0], 17)], dtype='float')
                    if len(frame_poses)
                    else np.zeros((17, 2), dtype='float')
                    for frame_poses
                    in get_keypoints_for_all_frames(input_file)], dtype='float')
            np.savez(output_path,
                     positions_2d={
                         viz_subject: {
                             viz_action: [
                                 final_poses
                             ]
                         }
                     },
                     metadata={
                         'layout_name': 'coco',
                         'num_joints': 17,
                         'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]],
                         'video_metadata': {
                             viz_subject: {
                                 'w': frame_width,
                                 'h': frame_height
                             }
                         }
                     })

    sys.path.append(video_pose_3d_path)
    from .run_video_pose_3d import process_custom_2d_to_3d_npz

    for root, dirs, files in os.walk(tmp_3d_results_path):
        if not dirs:
            pose_2d_npz_path = os.path.join(root, files[0])
            input_rel_path = os.path.relpath(root, tmp_3d_results_path)
            output_path = os.path.join(output_directory, input_rel_path, output_file_name)

            if not os.path.exists(output_path):
                pathlib.Path(os.path.dirname(output_path)).mkdir(parents=True)

            process_custom_2d_to_3d_npz(pose_2d_npz_path, video_pose_3d_path, viz_subject, viz_action, viz_export=output_path)

    if os.path.exists(tmp_3d_results_path):
        shutil.rmtree(tmp_3d_results_path)
