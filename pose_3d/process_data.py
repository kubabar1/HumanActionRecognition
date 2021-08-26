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
        print('generating 2D data ...')
        process_to_2d_single_person(input_directory, output_directory)
        print('2D data generated')
    print('generating 3D data ...')
    process_to_3d_single_person(input_directory, output_directory, frame_width, frame_height, video_pose_3d_path)
    print('3D data generated')


if __name__ == '__main__':
    main()
