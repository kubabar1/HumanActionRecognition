import argparse
from configparser import ConfigParser

from utils_3d.process_data_utils import process_to_2d_single_person, process_to_3d_single_person


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-pose-3d-path', help='Absolute path to VideoPose3D')
    parser.add_argument('--output-directory', help='Path to generated results output directory', type=str, default='./results/')
    parser.add_argument('--input-directory', help='Path to input data', required=True)
    parser.add_argument('--frame-width', help='Width of frame', required=True)
    parser.add_argument('--frame-height', help='Height of frame', required=True)
    parser.add_argument('--joints-count', help='Count of joints given as input', required=False, type=int, default=17)
    parser.add_argument('--joints-left', help='Joints from left part of silhouette', required=False, type=int, nargs='+',
                        default=[1, 3, 5, 7, 9, 11, 13, 15])
    parser.add_argument('--joints-right', help='Joints from right part of silhouette', required=False, type=int, nargs='+',
                        default=[2, 4, 6, 8, 10, 12, 14, 16])
    parser.add_argument('--generate-2D', help='Generate 2D coordinates too', default=False, action='store_true')

    args = parser.parse_args()
    video_pose_3d_path = args.video_pose_3d_path
    output_directory = args.output_directory
    input_directory = args.input_directory
    generate_2D = args.generate_2D
    frame_width = int(args.frame_width)
    frame_height = int(args.frame_height)
    joints_count = int(args.joints_count)
    joints_left = args.joints_left
    joints_right = args.joints_right

    if video_pose_3d_path is None:
        config = ConfigParser()
        config.read('../config.ini')
        video_pose_3d_path = config.get('main', 'VIDEO_POSE_3D_PATH')

    if generate_2D:
        print('generating 2D data ...')
        process_to_2d_single_person(input_directory, output_directory)
        print('2D data generated')
    print('generating 3D data ...')
    process_to_3d_single_person(video_pose_3d_path, input_directory, output_directory, frame_width, frame_height, joints_count, joints_left,
                                joints_right)
    print('3D data generated')


if __name__ == '__main__':
    main()
