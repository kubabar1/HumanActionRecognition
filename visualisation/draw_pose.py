import argparse
import os

import numpy as np

from utils.cv2_utils import get_frames_count, draw_points_multiple_poses, draw_points_single_pose, draw_3d_pose
from utils.hpe_results_utils import read_keypoints_all_poses, get_best_pose, read_keypoints_csv


def draw_pose(data_path, ground_truth_video, many_poses, draw_3d):
    if draw_3d:
        data = np.load(data_path)
        draw_3d_pose(data)
    else:
        frames_count = int(get_frames_count(ground_truth_video))
        if many_poses:
            data = read_keypoints_all_poses(data_path)
            draw_points_multiple_poses(ground_truth_video, data)
        else:
            if os.path.isdir(data_path):
                data = get_best_pose(read_keypoints_all_poses(data_path), frames_count)
            else:
                data = read_keypoints_csv(data_path)
            draw_points_single_pose(ground_truth_video, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', help='Path to data containing coordinates directory of "*.csv" files with poses for '
                                            '"many-poses", "*.csv" file with 2D single pose generated by MMPose with "hpe"module '
                                            'or "*.npy" file with 3D single pose generated bt VideoPose3D', required=True)
    parser.add_argument('--ground-truth-video', help='Path to data with ground truth video', default=None)
    parser.add_argument('--many-poses', help='Analyse many poses (only for 2D)', default=False, action='store_true')
    parser.add_argument('--draw-3d', help='Draw in 3D', default=False, action='store_true')

    args = parser.parse_args()
    data_path = args.data_path
    ground_truth_video = args.ground_truth_video
    many_poses = args.many_poses
    draw_3d = args.draw_3d

    if not draw_3d and ground_truth_video is None:
        raise Exception('For 2D draw ground truth video is required')

    draw_pose(data_path, ground_truth_video, many_poses, draw_3d)
