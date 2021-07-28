import argparse
from configparser import ConfigParser

from utils_3d.process_data_utils import process_to_2d_single_person, process_to_3d_single_person


# (berkeley_frame_width = 640, berkeley_frame_height = 480) (ntu_rgbd_frame_width = 1920, ntu_rgbd_frame_height = 1080)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-pose-3d-path', help='Absolute path to VideoPose3D')
    parser.add_argument('--output-directory', help='Path to generated results output directory', type=str, default='./results/')
    parser.add_argument('--input-directory', help='Path to input data', required=True)
    parser.add_argument('--frame-width', help='Width of frame', required=True)
    parser.add_argument('--frame-height', help='Height of frame', required=True)
    parser.add_argument('--generate-2D', help='Generate 2D coordinates too', default=False, action='store_true')

    args = parser.parse_args()
    video_pose_3d_path = args.video_pose_3d_path
    output_directory = args.output_directory
    input_directory = args.input_directory
    generate_2D = args.generate_2D
    frame_width = int(args.frame_width)
    frame_height = int(args.frame_height)

    if video_pose_3d_path is None:
        config = ConfigParser()
        config.read('../config.ini')
        video_pose_3d_path = config.get('main', 'VIDEO_POSE_3D_PATH')

    if generate_2D:
        process_to_2d_single_person(input_directory, output_directory)
    process_to_3d_single_person(input_directory, output_directory, frame_width, frame_height, video_pose_3d_path)

    # vid_ref = '/home/kuba/workspace/human_action_recognition/datasets/BerkeleyMHAD/Camera/Cluster01/Cam01/S01/A02/R01.mp4'
    # poses_keypoints = np.load(
    #     '/home/kuba/workspace/human_action_recognition/HumanActionRecognition/datasets/berkeley_mhad/3d/s01/a02/r01/3d_coordinates.npy')
    # cap = cv2.VideoCapture(vid_ref)
    # frame_id = 0
    # stop = False
    # while True:
    #     if not stop:
    #         ret, frame = cap.read()
    #         if ret:
    #             for i in range(17):
    #                 x = int(float(poses_keypoints[frame_id][i][0] * (berkeley_frame_width / 2) + berkeley_frame_width / 2))
    #                 y = int(float(poses_keypoints[frame_id][i][1] * (berkeley_frame_height / 2) + berkeley_frame_height / 2))
    #                 cv2.circle(frame, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
    #             frame = cv2.resize(frame, (500, 500))
    #             cv2.imshow('frame', frame)
    #             frame_id += 1
    #         else:
    #             break
    #     k = cv2.waitKey(25) & 0xFF
    #     if k == ord('q'):
    #         break
    #     if k == ord(' '):
    #         stop = not stop
    # cap.release()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
