import os
import time

import cv2
from PIL import Image
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import (init_pose_model, inference_top_down_pose_model, inference_bottom_up_pose_model, vis_pose_result)

from .pose_estimation_utils import create_output_directories, process_mmdet_results, save_keypoints_csv, save_bounding_box_csv, \
    print_progress, purge_files


def estimate_pose(det_config, det_checkpoint, pose_config, pose_checkpoint, input_path, output_path, is_video=True,
                  device='cuda:0', return_heatmap=False, save_keypoints=True, save_bounding_boxes=True, save_out_video=True,
                  kpt_thr=0.3, bbox_thr=0.3, model_type='top_down', output_layer_names=None):
    video_writer = None
    cap = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    model_name = os.path.splitext(os.path.basename(pose_config))[0]

    out_video_root, output_keypoints_root, output_bbox_root = None, None, None
    if save_keypoints or save_bounding_boxes or save_out_video:
        out_video_root, output_keypoints_root, output_bbox_root = create_output_directories(output_path, model_name,
                                                                                            save_out_video,
                                                                                            save_keypoints,
                                                                                            save_bounding_boxes)

    det_model = init_detector(det_config, det_checkpoint, device=device)
    pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)
    dataset = pose_model.cfg.data['test']['type']

    if is_video:
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        if save_keypoints:
            purge_files(output_keypoints_root, input_name)
        if save_bounding_boxes:
            purge_files(output_bbox_root, input_name)
        if save_out_video:
            purge_files(out_video_root, input_name)

        cap = cv2.VideoCapture(input_path)
        if save_out_video and out_video_root is not None:
            fps = cap.get(cv2.CAP_PROP_FPS)
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            video_writer = cv2.VideoWriter(os.path.join(out_video_root, 'vis_' + os.path.basename(input_path)), fourcc, fps,
                                           size)
        estimate_pose_movie(cap, input_path, det_model, pose_model, video_writer, dataset, out_video_root,
                            output_keypoints_root, output_bbox_root, kpt_thr, bbox_thr, return_heatmap, model_type,
                            output_layer_names)
    else:
        inputs_paths = sorted([os.path.join(input_path, f) for f in os.listdir(input_path) if
                               os.path.isfile(os.path.join(input_path, f))])
        input_name = os.path.splitext(inputs_paths[0].split(os.path.sep)[-1])[0]

        if save_keypoints:
            purge_files(output_keypoints_root, input_name)
        if save_bounding_boxes:
            purge_files(output_bbox_root, input_name)
        if save_out_video:
            purge_files(out_video_root, input_name)

        if save_out_video and out_video_root is not None:
            fps = 20
            size = cv2.imread(inputs_paths[0]).shape[:2]
            size = (size[1], size[0])
            output_video_name = input_name + '.mp4'
            video_writer = cv2.VideoWriter(os.path.join(out_video_root, output_video_name), fourcc, fps, size)
        estimate_pose_images_sequence(inputs_paths, det_model, pose_model, video_writer, dataset, input_name, out_video_root,
                                      output_keypoints_root, output_bbox_root, kpt_thr, bbox_thr, return_heatmap, model_type,
                                      output_layer_names)

    if is_video and cap is not None:
        cap.release()
    if save_out_video and video_writer is not None:
        video_writer.release()


def estimate_pose_movie(cap, input_name, det_model, pose_model, video_writer, dataset, output_video_root, output_keypoints_root,
                        output_bbox_root, kpt_thr, bbox_thr, return_heatmap, model_type, output_layer_names):
    frame = 0
    start_time = time.time()
    all_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        flag, img = cap.read()
        if not flag:
            break
        estimate_pose_common(img, det_model, pose_model, video_writer, frame, start_time, input_name, dataset, output_video_root,
                             output_keypoints_root, output_bbox_root, all_frames_count, bbox_thr, return_heatmap,
                             output_layer_names, kpt_thr, model_type)
        frame += 1


def estimate_pose_images_sequence(images_sequence_array, det_model, pose_model, video_writer, dataset, input_name,
                                  output_video_root, output_keypoints_root, output_bbox_root, kpt_thr, bbox_thr, return_heatmap,
                                  model_type, output_layer_names):
    frame = 0
    start_time = time.time()
    all_frames_count = len(images_sequence_array)

    for img_path in images_sequence_array:
        try:
            img = Image.open(img_path)
            img.verify()
            estimate_pose_common(cv2.imread(img_path), det_model, pose_model, video_writer, frame, start_time, input_name,
                                 dataset, output_video_root, output_keypoints_root, output_bbox_root, all_frames_count, bbox_thr,
                                 return_heatmap, output_layer_names, kpt_thr, model_type)
        except (IOError, SyntaxError):
            print('Bad file:', img_path)
        frame += 1


def estimate_pose_common(img, det_model, pose_model, video_writer, frame, start_time, input_name, dataset, output_video_root,
                         output_keypoints_root, output_bbox_root, all_frames_count, bbox_thr, return_heatmap,
                         output_layer_names, kpt_thr, model_type):
    person_bboxes = None
    if model_type == 'top_down':
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, img)

        # keep the person class bounding boxes.
        person_bboxes = process_mmdet_results(mmdet_results)

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_bboxes,
            bbox_thr=bbox_thr,
            format='xyxy',
            dataset=dataset,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
    else:
        pose_results, returned_outputs = inference_bottom_up_pose_model(
            pose_model,
            img,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

    if output_video_root is not None:
        vis_img = vis_pose_result(
            pose_model,
            img,
            pose_results,
            dataset=dataset,
            kpt_score_thr=kpt_thr,
            show=False)
        video_writer.write(vis_img)

    if output_keypoints_root is not None:
        save_keypoints_csv(pose_results, output_keypoints_root, frame, input_name)

    if output_bbox_root is not None and person_bboxes is not None and model_type == 'top_down':
        save_bounding_box_csv(person_bboxes, output_bbox_root, frame, input_name)

    step_time_estimate = 10

    if not (frame % step_time_estimate):
        print_progress(frame, start_time, all_frames_count)
