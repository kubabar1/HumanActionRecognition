from har_pose_3d.run_video_pose_3d import process_2d_to_3d, load_model
import numpy as np


def main():
    video_pose_3d_path = '/home/kuba/workspace/human_action_recognition/VideoPose3D'
    pt = 'VideoPose3D.npz'
    keypoints = np.array(np.load(pt, allow_pickle=True)['positions_2d'].item()['VideoPose3D']['custom'][0])
    joints_count = 17

    joints_left = [1, 3, 5, 7, 9, 11, 13, 15]
    joints_right = [2, 4, 6, 8, 10, 12, 14, 16]

    frame_width = 1080
    frame_height = 1920

    model_pos = load_model(video_pose_3d_path, joints_count)
    prediction = process_2d_to_3d(video_pose_3d_path, keypoints, model_pos, joints_left, joints_right, frame_width, frame_height)

    np.save('qwe', prediction)


if __name__ == '__main__':
    main()
