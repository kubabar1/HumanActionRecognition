import os

import torch
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_pose_model, inference_top_down_pose_model, inference_bottom_up_pose_model

from .hpe_utils.top_down_configs import top_down_configs


def load_models(mmpose_path, hpe_method='res152_coco_384x288', device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                det_checkpoint=None, det_config=None, pose_checkpoint=None, pose_config=None):
    """
    Load MMDetection and MMPose models

    :param mmpose_path: path to MMPose
    :param hpe_method: HPE method (for human pose estimation) (by default 'res152_coco_384x288')
    :param device: used device (by default 'cuda:0' if is available, else 'cpu')
    :param det_checkpoint: checkpoint for detection model (by default used
        'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')
    :param det_config: config of detection model (by default used
        os.path.join(mmpose_path, 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_1x_coco.py'))
    :param pose_checkpoint: checkpoint for hpe model (by default used
        'https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_384x288_dark-d3b8ebd7_20210203.pth')
    :param pose_config: config of hpe model (by default used
        os.path.join(mmpose_path, 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_1x_coco.py'))
    :return: MMDetection and MMPose models
    """
    if det_config is None:
        det_config = os.path.join(mmpose_path, 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_1x_coco.py')
    if det_checkpoint is None:
        det_checkpoint = 'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco' \
                         '/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    det_model = init_detector(det_config, det_checkpoint, device=device)

    p_config = top_down_configs(mmpose_path)[hpe_method]
    if pose_config is None:
        pose_config = p_config[0]
    if pose_checkpoint is None:
        pose_checkpoint = p_config[1]
    pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)

    return det_model, pose_model


def estimate_pose(frame, pose_model, det_model=None, bbox_thr=0.3):
    """
    Estimate keypoints for given frame

    :param frame: input frame for pose estimation
    :param pose_model: HPE model loaded using 'load_models' function
    :param det_model: human silhouette detector model loaded using 'load_models' function (if not given, then bottom-up method flow is used)
    :param bbox_thr: threshold for human silhouette detections (using human detector)
    :return: estimated keypoints for given frame
    """
    if det_model is not None:
        mmdet_results = inference_detector(det_model, frame)
        person_bboxes = process_mmdet_results(mmdet_results)
        dataset = pose_model.cfg.data['test']['type']
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            frame,
            person_bboxes,
            bbox_thr=bbox_thr,
            format='xyxy',
            dataset=dataset,
            return_heatmap=False,
            outputs=None)
    else:
        pose_results, _ = inference_bottom_up_pose_model(
            pose_model,
            frame,
            return_heatmap=False,
            outputs=None)
    return pose_results


def process_mmdet_results(mmdet_results, cat_id=0):
    """Process mmdet results, and return a list of bboxes.

    :param mmdet_results:
    :param cat_id: category id (default: 0 for human)
    :return: a list of detected bounding boxes
    """
    if isinstance(mmdet_results, tuple):
        det_results = mmdet_results[0]
    else:
        det_results = mmdet_results

    bboxes = det_results[cat_id]

    person_results = []
    for bbox in bboxes:
        person = {'bbox': bbox}
        person_results.append(person)

    return person_results
