import os
import cv2
import sys
import pathlib
import csv
import shutil
from configparser import ConfigParser
import numpy as np
import importlib

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
    tmp_3d_results_path = os.path.join(results_path, 'tmp')
    if os.path.exists(results_path):
        shutil.rmtree(results_path)

    viz_subject = 'VideoPose3D'
    viz_action = 'custom'

    output_file_name = '3d_coordinates'

    for subject_id, subject_path in enumerate(sorted([f.path for f in os.scandir(silhouettes_path)])):
        for action_dir_path in sorted([f.path for f in os.scandir(subject_path)]):
            for repetition_id, repetition_path in enumerate(sorted([f.path for f in os.scandir(action_dir_path)])):
                pr = os.path.join(repetition_path, 'pose_results')
                res = os.path.join(tmp_3d_results_path, '/'.join(repetition_path.split('/')[1:]), viz_subject)
                if not os.path.exists(res):
                    pathlib.Path(os.path.dirname(res)).mkdir(parents=True)
                final_poses = np.array(
                    [
                        np.array([np.array([i[0], i[1]], dtype='float32')
                                  for i in np.array_split(frame_poses[0], 17)], dtype='float32')
                        if len(frame_poses)
                        else np.zeros((17, 2), dtype='float32')
                        for frame_poses
                        in get_keypoints_for_all_frames(pr)], dtype='float32')
                np.savez(res,
                         positions_2d={
                             viz_subject: {
                                 viz_action: [
                                     final_poses
                                 ]
                             }
                         }, metadata={
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
    from run_video_pose_3d import process_custom_2d_to_3d_npz

    for subject_path, subject_name in sorted([(f.path, f.name) for f in os.scandir(tmp_3d_results_path)]):
        for action_dir_path, action_name in sorted([(f.path, f.name) for f in os.scandir(subject_path)]):
            for repetition_path, repetition_name in sorted([(f.path, f.name) for f in os.scandir(action_dir_path)]):
                pose_2d_npz_path = os.path.join(repetition_path, viz_subject + '.npz')
                output_path = os.path.join(results_path, subject_name, action_name, repetition_name, output_file_name)
                if not os.path.exists(output_path):
                    pathlib.Path(os.path.dirname(output_path)).mkdir(parents=True)
                # print('#######################################################################')
                # print(pose_2d_npz_path)
                # print(viz_subject)
                # print(viz_action)
                # print(output_path)
                # print('#######################################################################')
                process_custom_2d_to_3d_npz(pose_2d_npz_path, video_pose_3d_path, viz_subject, viz_action, viz_export=output_path)

    if os.path.exists(tmp_3d_results_path):
        shutil.rmtree(tmp_3d_results_path)


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
    # process_berkeley_mhad_to_3d_single_person(silhouettes_path, results_path_3d, berkeley_frame_width, berkeley_frame_height,
    #                                           video_pose_3d_path)
    vid_ref = '/home/kuba/workspace/human_action_recognition/datasets/BerkeleyMHAD/Camera/Cluster01/Cam01/S01/A02/R01.mp4'
    poses_keypoints = np.load(
        '/home/kuba/workspace/human_action_recognition/HumanActionRecognition/datasets/berkeley_mhad/3d/s01/a02/r01/3d_coordinates.npy')
    cap = cv2.VideoCapture(vid_ref)
    frame_id = 0
    stop = False
    while True:
        if not stop:
            ret, frame = cap.read()
            if ret:
                for i in range(17):
                    x = int(float(poses_keypoints[frame_id][i][0] * (berkeley_frame_width / 2) + berkeley_frame_width / 2))
                    y = int(float(poses_keypoints[frame_id][i][1] * (berkeley_frame_height / 2) + berkeley_frame_height / 2))
                    cv2.circle(frame, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
                frame = cv2.resize(frame, (500, 500))
                cv2.imshow('frame', frame)
                frame_id += 1
            else:
                break
        k = cv2.waitKey(25) & 0xFF
        if k == ord('q'):
            break
        if k == ord(' '):
            stop = not stop
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
