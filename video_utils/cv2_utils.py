import cv2
import numpy as np


def get_frames_count(video_path):
    return cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT)


def get_video_frame_size(video_path):
    height = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_WIDTH)
    return height, width


def draw_points(true_video_path, predicted_kpts, predicted_bbox, analyzed_method_name, min_acc=0.0, pause_l=0.05):
    cap = cv2.VideoCapture(true_video_path)
    stop = False
    skip = False
    frame_id = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    lineType = 1

    while True:
        if not stop:
            ret, frame = cap.read()
            if ret:
                for pose_kpts in [predicted_kpts[frame_id]]:
                    kpts = np.array_split(pose_kpts, 17)
                    for k in kpts:
                        if len(k) and k[2] >= min_acc:
                            cv2.circle(frame, (int(k[0]), int(k[1])), 1, (0, 0, 255), 5)
                    cv2.putText(frame,
                                analyzed_method_name,
                                (10, 30),
                                font,
                                fontScale,
                                (0, 0, 255),
                                lineType)
                    cv2.putText(frame,
                                'Frame: ' + str(frame_id),
                                (10, 50),
                                font,
                                fontScale,
                                (0, 255, 0),
                                lineType)
                    cv2.putText(frame,
                                'Mean accuracy: ' + str(sum_acc_all(pose_kpts) / 17),
                                (10, 70),
                                font,
                                fontScale,
                                (0, 255, 0),
                                lineType)
                for pose_bbox in [predicted_bbox[frame_id]]:
                    if len(pose_bbox) == 5:
                        x1, y1, x2, y2, acc = pose_bbox
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        if acc >= min_acc:
                            cv2.line(frame, (x1, y1), (x2, y1), (0, 255, 0), thickness=1)
                            cv2.line(frame, (x1, y2), (x2, y2), (0, 255, 0), thickness=1)
                            cv2.line(frame, (x1, y1), (x1, y2), (0, 255, 0), thickness=1)
                            cv2.line(frame, (x2, y1), (x2, y2), (0, 255, 0), thickness=1)
                cv2.imshow(os.path.basename(true_video_path), frame)
                frame_id += 1
            else:
                break
            time.sleep(pause_l)
        ky = cv2.waitKey(25) & 0xFF
        if skip:
            skip = False
            stop = not stop
        if ky == ord('q'):
            break
        if ky == ord(' '):
            stop = not stop
        if ky == 9:
            stop = not stop
            skip = True
    cap.release()
    cv2.destroyAllWindows()


def draw_points_multiple_poses(true_video_path, predicted_kpts, predicted_bbox, analyzed_method_name, min_acc=0.0, pause_l=0.05,
                               show_bbox=True, show_kpts=True, max_pose_count=None, show_text=True):
    cap = cv2.VideoCapture(true_video_path)
    stop = False
    skip = False
    frame_id = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    lineType = 2

    max_pred_pose_cnt = max([len(i) for i in predicted_kpts])
    random_colors = [list(np.random.random(size=3) * 256) for i in range(max_pred_pose_cnt)]

    while True:
        if not stop:
            ret, frame = cap.read()
            if ret:
                tmp = order_match_array_by_acc(predicted_kpts, len(predicted_kpts))
                if tmp and show_kpts:
                    for pose_id, pose in enumerate(tmp[frame_id] if not max_pose_count else tmp[frame_id][:max_pose_count]):
                        color = random_colors[pose_id]
                        kpts = np.array_split(pose, 17)
                        for k in kpts:
                            if len(k) and k[2] >= min_acc:
                                cv2.circle(frame, (int(k[0]), int(k[1])), 1, color, 5)
                                if show_text:
                                    cv2.putText(frame,
                                                'Pose {}: {}'.format(str(pose_id), sum_acc_all(pose) / 17),
                                                (10, 70 + 20 * pose_id),
                                                font,
                                                fontScale,
                                                color,
                                                lineType)
                tmp2 = order_match_bbox_by_acc(predicted_bbox, len(predicted_bbox))
                if tmp2 and show_bbox:
                    for pose_bbox_id, pose_bbox in enumerate(
                            tmp2[frame_id] if not max_pose_count else tmp2[frame_id][:max_pose_count]):
                        if len(pose_bbox) == 5:
                            x1, y1, x2, y2, acc = pose_bbox
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            if acc >= min_acc:
                                cv2.line(frame, (x1, y1), (x2, y1), (0, 255, 0), thickness=1)
                                cv2.line(frame, (x1, y2), (x2, y2), (0, 255, 0), thickness=1)
                                cv2.line(frame, (x1, y1), (x1, y2), (0, 255, 0), thickness=1)
                                cv2.line(frame, (x2, y1), (x2, y2), (0, 255, 0), thickness=1)
                if show_text:
                    cv2.putText(frame,
                                analyzed_method_name,
                                (10, 30),
                                font,
                                fontScale,
                                (0, 0, 255),
                                lineType)
                    cv2.putText(frame,
                                'Frame: ' + str(frame_id),
                                (10, 50),
                                font,
                                fontScale,
                                (0, 255, 0),
                                lineType)
                cv2.imshow('frame', frame)
                frame_id += 1
            else:
                break
            time.sleep(pause_l)
        ky = cv2.waitKey(25) & 0xFF
        if skip:
            skip = False
            stop = not stop
        if ky == ord('q'):
            break
        if ky == ord(' '):
            stop = not stop
        if ky == 9:
            stop = not stop
            skip = True
    cap.release()
    cv2.destroyAllWindows()
