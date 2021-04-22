import pathlib

import numpy as np
import os
import csv


def main():
    coordinates_path = 'coordinates'
    silhouettes_path = 'unprocessed'
    for subject_id, subject_path in enumerate([f.path for f in os.scandir(silhouettes_path)]):
        for action_dir_path in [f.path for f in os.scandir(subject_path)]:
            for repetition_id, repetition_path in enumerate([f.path for f in os.scandir(action_dir_path)]):
                pr = os.path.join(repetition_path, 'pose_results')
                res = os.path.join(coordinates_path, '/'.join(repetition_path.split('/')[1:]), 'x.csv')
                pathlib.Path(os.path.dirname(res)).mkdir(parents=True)
                final_pose = get_best_poses_array(pr)
                with open(res, "w+") as my_csv:
                    csvWriter = csv.writer(my_csv, delimiter=',')
                    csvWriter.writerows(final_pose)


def sum_acc_all(pose):
    return sum(pose[2::3])


def get_frame_with_better_acc(poses_array, frame_id, joints_count=17):
    best_pose = np.zeros(17 * 3)  # TODO - ensure what to do if point was not estimated
    for pose in poses_array:
        if frame_id in pose:
            best_pose = np.array(pose[frame_id], dtype='float') if \
                sum_acc_all(np.array(pose[frame_id], dtype='float')) >= sum_acc_all(best_pose) else best_pose
    return [v for i, v in enumerate(best_pose) if (i + 1) % 3]


def get_best_poses_array(poses_directory):
    last_frame_id = int(
        max([get_last_frame_id(os.path.join(poses_directory, pf)) for pf in os.listdir(poses_directory)]))
    poses_array = []
    for pose_file in os.listdir(poses_directory):
        tmp_pose = {}
        pose_path = os.path.join(poses_directory, pose_file)
        with open(pose_path, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            for i, row in enumerate(reader):
                tmp_pose[i] = row[1:]
        poses_array.append(tmp_pose)

    final_pose = []
    for frame_id in range(last_frame_id):
        final_pose.append(get_frame_with_better_acc(poses_array, frame_id))
    final_pose = np.array(final_pose)
    return final_pose


def get_csv_rows_count(csv_reader):
    return sum(1 for _ in csv_reader)


def get_last_frame_id(csv_path):
    csv_reader = csv.reader(open(csv_path, 'r'), delimiter=',')
    last_row = None
    for last_row in csv_reader:
        pass
    return last_row[0] if last_row else None


if __name__ == '__main__':
    main()
