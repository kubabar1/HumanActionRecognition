import argparse
import os
from configparser import ConfigParser

from hpe_utils.pose_estimation import estimate_pose
from hpe_utils.top_down_configs import top_down_configs


def main():
    parser = argparse.ArgumentParser('simple_example')
    parser.add_argument('--mmpose-path', help='Absolute path to MMPose')
    parser.add_argument('--hpe-method', help='MMPose method to estimate human pose', type=str, default='res152_coco_384x288')
    parser.add_argument('--output-directory', help='Path to generated results output directory', type=str, default='./results/')
    parser.add_argument('--dataset-path', help='Path to input dataset', required=True)
    parser.add_argument('--video', default=False, action='store_true')

    args = parser.parse_args()
    mmpose_path = args.mmpose_path
    hpe_method = args.hpe_method
    output_directory = args.output_directory
    dataset_path = args.dataset_path
    is_video = args.video

    if mmpose_path is None:
        config = ConfigParser()
        config.read('../config.ini')
        mmpose_path = config.get('main', 'MMPOSE_PATH')

    det_config = os.path.join(mmpose_path, 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_1x_coco.py')
    det_checkpoint = 'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

    p_config = top_down_configs(mmpose_path)[hpe_method]
    pose_config = p_config[0]
    pose_checkpoint = p_config[1]

    for root, dirs, files in os.walk(dataset_path):
        if not dirs:
            input_rel_path = os.path.relpath(root, dataset_path)
            output_path = os.path.join(output_directory, input_rel_path)
            if is_video:
                for input_video in files:
                    estimate_pose(det_config, det_checkpoint, pose_config, pose_checkpoint, os.path.join(root, input_video),
                                  output_path, is_video=is_video)
            else:
                estimate_pose(det_config, det_checkpoint, pose_config, pose_checkpoint, root, output_path, is_video=is_video)


if __name__ == '__main__':
    main()
