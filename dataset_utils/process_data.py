import os
import sys
import pathlib
import csv
import shutil
from configparser import ConfigParser
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.curdir, '..')))

from hpe_utils.mmpose_results_utils import get_keypoints_for_all_frames, get_bbox_for_all_frames


def process_berkeley_mhad_to_2d_single_person(silhouettes_path, results_path):
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    for subject_id, subject_path in enumerate(sorted([f.path for f in os.scandir(silhouettes_path)])):
        for action_dir_path in sorted([f.path for f in os.scandir(subject_path)]):
            for repetition_id, repetition_path in enumerate(sorted([f.path for f in os.scandir(action_dir_path)])):
                pr = os.path.join(repetition_path, 'pose_results')
                res = os.path.join(results_path, '/'.join(repetition_path.split('/')[1:]), 'x.csv')
                if not os.path.exists(res):
                    pathlib.Path(os.path.dirname(res)).mkdir(parents=True)
                final_pose = [poses[0] if len(poses) else [] for poses in get_keypoints_for_all_frames(pr)]
                with open(res, "w+") as my_csv:
                    csvWriter = csv.writer(my_csv, delimiter=',')
                    csvWriter.writerows(final_pose)


def process_berkeley_mhad_to_3d_single_person(silhouettes_path, results_path, frame_width, frame_height, video_pose_3d_path):
    sys.path.append(video_pose_3d_path)
    from run_video_pose_3d import process_custom_2d_to_3d
    # from run_video_pose_3d import process_custom_2d_npz_to_3d
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    keypoints_npz_path = '/home/kuba/workspace/human_action_recognition/VideoPose3D/data/data_2d_custom_kinect2.npz'
    process_custom_2d_to_3d(keypoints_npz_path, video_pose_3d_path,
                            '/home/kuba/workspace/human_action_recognition/VideoPose3D/kinect2', 'kinect2', 'custom', viz_output=None)
    # for subject_id, subject_path in enumerate(sorted([f.path for f in os.scandir(silhouettes_path)])):
    #     for action_dir_path in sorted([f.path for f in os.scandir(subject_path)]):
    #         for repetition_id, repetition_path in enumerate(sorted([f.path for f in os.scandir(action_dir_path)])):
    #             pr = os.path.join(repetition_path, 'pose_results')
    #             # br = os.path.join(repetition_path, 'bbox')
    #             res = os.path.join(results_path, '/'.join(repetition_path.split('/')[1:]), 'VideoPose3D')
    #             if not os.path.exists(res):
    #                 pathlib.Path(os.path.dirname(res)).mkdir(parents=True)
    #             final_poses = np.array(
    #                 [
    #                     np.array([np.array([i[0], i[1]], dtype='float32')
    #                               for i in np.array_split(frame_poses[0], 17)], dtype='float32')
    #                     if len(frame_poses)
    #                     else np.zeros((17, 2), dtype='float32')
    #                     for frame_poses
    #                     in get_keypoints_for_all_frames(pr)], dtype='float32')
    #             # final_bboxes = [poses[0] if len(poses) else [] for poses in get_bbox_for_all_frames(br)]
    #             np.savez(res,
    #                      positions_2d={
    #                          'VideoPose3D': {
    #                              'custom': [
    #                                  final_poses
    #                              ]
    #                          }
    #                      }, metadata={
    #                     'layout_name': 'coco',
    #                     'num_joints': 17,
    #                     'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]],
    #                     'video_metadata': {
    #                         'VideoPose3D': {
    #                             'w': frame_width,
    #                             'h': frame_height
    #                         }
    #                     }
    #                 })
    # qwe = '/home/kuba/workspace/human_action_recognition/HumanActionRecognition/har_implementations/pose_based_har_framework/data/videopose3d/s01/a01/r01/VideoPose3D.npz'
    # process_custom_2d_npz_to_3d('checkpoint', video_pose_3d_path, qwe)


def main():
    results_path_2d = '../datasets/berkeley_mhad/2d'
    results_path_3d = '../datasets/berkeley_mhad/3d'
    silhouettes_path = 'unprocessed'
    config = ConfigParser()
    config.read('../config.ini')
    video_pose_3d_path = config.get('main', 'VIDEO_POSE_3D_PATH')
    berkeley_frame_width = 640
    berkeley_frame_height = 480

    # process_berkeley_mhad_to_2d_single_person(silhouettes_path, results_path_2d)
    process_berkeley_mhad_to_3d_single_person(silhouettes_path, results_path_3d, berkeley_frame_width, berkeley_frame_height,
                                              video_pose_3d_path)

    # pr = '/home/kuba/workspace/pose_estimation/vis_results/kinect/top_down/hrnet_w48_coco_384x288_dark/pose_results'
    # final_poses = np.array(
    #     [
    #         np.array([np.array([i[0], i[1]], dtype='float32')
    #                   for i in np.array_split(frame_poses[0], 17)], dtype='float32')
    #         if len(frame_poses)
    #         else np.zeros((17, 2), dtype='float32')
    #         for frame_poses
    #         in get_keypoints_for_all_frames(pr)], dtype='float32')
    # # final_bboxes = [poses[0] if len(poses) else [] for poses in get_bbox_for_all_frames(br)]
    # np.savez('data_2d_custom_kinect2',
    #          positions_2d={
    #              'kinect2': {
    #                  'custom': [
    #                      final_poses
    #                  ]
    #              }
    #          }, metadata={
    #         'layout_name': 'coco',
    #         'num_joints': 17,
    #         'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]],
    #         'video_metadata': {
    #             'kinect2': {
    #                 'w': berkeley_frame_width,
    #                 'h': berkeley_frame_height
    #             }
    #         }
    #     })


if __name__ == '__main__':
    main()
