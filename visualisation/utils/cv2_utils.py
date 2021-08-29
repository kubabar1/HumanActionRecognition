import os
import time

import cv2
import numpy as np
import scipy.spatial
from matplotlib import pyplot as plt


def get_frames_count(video_path):
    return cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT)


def get_video_frame_size(video_path):
    height = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_WIDTH)
    return height, width


def draw_points_single_pose(true_video_path, predicted_kpts, min_acc=0, pause_l=0.05, show_text=True):
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
                pose_kpts = predicted_kpts[frame_id]
                for ki in pose_kpts:
                    if len(ki) and ki[2] >= min_acc:
                        cv2.circle(frame, (int(ki[0]), int(ki[1])), 1, (0, 0, 255), 5)
                if show_text:
                    cv2.putText(frame,
                                'Frame: ' + str(frame_id),
                                (10, 50),
                                font,
                                fontScale,
                                (0, 255, 0),
                                lineType)
                    cv2.putText(frame,
                                'Mean accuracy: ' + str(np.sum([i[2] for i in pose_kpts]) / 17),
                                (10, 70),
                                font,
                                fontScale,
                                (0, 255, 0),
                                lineType)
                # if predicted_bbox:
                #     for pose_bbox in [predicted_bbox[frame_id]]:
                #         if len(pose_bbox) == 5:
                #             x1, y1, x2, y2, acc = pose_bbox
                #             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                #             if acc >= min_acc:
                #                 cv2.line(frame, (x1, y1), (x2, y1), (0, 255, 0), thickness=1)
                #                 cv2.line(frame, (x1, y2), (x2, y2), (0, 255, 0), thickness=1)
                #                 cv2.line(frame, (x1, y1), (x1, y2), (0, 255, 0), thickness=1)
                #                 cv2.line(frame, (x2, y1), (x2, y2), (0, 255, 0), thickness=1)
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


def draw_points_multiple_poses(true_video_path, predicted_poses, min_acc=0.0, pause_l=0.05, show_text=True):
    cap = cv2.VideoCapture(true_video_path)
    stop = False
    skip = False
    frame_id = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    lineType = 2

    max_pred_pose_cnt = max([len(i) for i in predicted_poses])
    random_colors = [list(np.random.random(size=3) * 256) for i in range(max_pred_pose_cnt)]

    while True:
        if not stop:
            ret, frame = cap.read()
            if ret:
                for pose_id, pose in enumerate(predicted_poses):
                    color = random_colors[pose_id]
                    if frame_id in pose:
                        kpts = pose[frame_id]
                        for k in kpts:
                            if len(k) and k[2] >= min_acc:
                                cv2.circle(frame, (int(k[0]), int(k[1])), 1, color, 5)
                                if show_text:
                                    cv2.putText(frame,
                                                'Pose {}: {}'.format(str(pose_id), np.mean([i[2] for i in kpts])),
                                                (10, 70 + 20 * pose_id),
                                                font,
                                                fontScale,
                                                color,
                                                lineType)
                # if show_bbox:
                #     tmp2 = order_match_bbox_by_acc(predicted_bbox, len(predicted_bbox))
                #     if tmp2:
                #         for pose_bbox_id, pose_bbox in enumerate(
                #                 tmp2[frame_id] if not max_pose_count else tmp2[frame_id][:max_pose_count]):
                #             if len(pose_bbox) == 5:
                #                 x1, y1, x2, y2, acc = pose_bbox
                #                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                #                 if acc >= min_acc:
                #                     cv2.line(frame, (x1, y1), (x2, y1), (0, 255, 0), thickness=1)
                #                     cv2.line(frame, (x1, y2), (x2, y2), (0, 255, 0), thickness=1)
                #                     cv2.line(frame, (x1, y1), (x1, y2), (0, 255, 0), thickness=1)
                #                     cv2.line(frame, (x2, y1), (x2, y2), (0, 255, 0), thickness=1)
                if show_text:
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


def draw_3d_pose(data_3d):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    line0, = ax.plot(np.array([data_3d[0][15][0], data_3d[0][16][0]]),
                     np.array([data_3d[0][15][1], data_3d[0][16][1]]),
                     np.array([data_3d[0][15][2], data_3d[0][16][2]]))
    line1, = ax.plot(np.array([data_3d[0][15][0], data_3d[0][14][0]]),
                     np.array([data_3d[0][15][1], data_3d[0][14][1]]),
                     np.array([data_3d[0][15][2], data_3d[0][14][2]]))

    line2, = ax.plot(np.array([data_3d[0][13][0], data_3d[0][12][0]]),
                     np.array([data_3d[0][13][1], data_3d[0][12][1]]),
                     np.array([data_3d[0][13][2], data_3d[0][12][2]]))
    line3, = ax.plot(np.array([data_3d[0][11][0], data_3d[0][12][0]]),
                     np.array([data_3d[0][11][1], data_3d[0][12][1]]),
                     np.array([data_3d[0][11][2], data_3d[0][12][2]]))

    line4, = ax.plot(np.array([data_3d[0][11][0], data_3d[0][14][0]]),
                     np.array([data_3d[0][11][1], data_3d[0][14][1]]),
                     np.array([data_3d[0][11][2], data_3d[0][14][2]]))

    line5, = ax.plot(np.array([data_3d[0][11][0], data_3d[0][4][0]]),
                     np.array([data_3d[0][11][1], data_3d[0][4][1]]),
                     np.array([data_3d[0][11][2], data_3d[0][4][2]]))
    line6, = ax.plot(np.array([data_3d[0][1][0], data_3d[0][14][0]]),
                     np.array([data_3d[0][1][1], data_3d[0][14][1]]),
                     np.array([data_3d[0][1][2], data_3d[0][14][2]]))

    line7, = ax.plot(np.array([data_3d[0][1][0], data_3d[0][4][0]]),
                     np.array([data_3d[0][1][1], data_3d[0][4][1]]),
                     np.array([data_3d[0][1][2], data_3d[0][4][2]]))

    line8, = ax.plot(np.array([data_3d[0][1][0], data_3d[0][2][0]]),
                     np.array([data_3d[0][1][1], data_3d[0][2][1]]),
                     np.array([data_3d[0][1][2], data_3d[0][2][2]]))
    line9, = ax.plot(np.array([data_3d[0][2][0], data_3d[0][3][0]]),
                     np.array([data_3d[0][2][1], data_3d[0][3][1]]),
                     np.array([data_3d[0][2][2], data_3d[0][3][2]]))

    line10, = ax.plot(np.array([data_3d[0][5][0], data_3d[0][6][0]]),
                      np.array([data_3d[0][5][1], data_3d[0][6][1]]),
                      np.array([data_3d[0][5][2], data_3d[0][6][2]]))
    line11, = ax.plot(np.array([data_3d[0][5][0], data_3d[0][4][0]]),
                      np.array([data_3d[0][5][1], data_3d[0][4][1]]),
                      np.array([data_3d[0][5][2], data_3d[0][4][2]]))

    ax.view_init(90, 90)
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    for f in range(1, len(data_3d)):
        # data_3d = rotate(data_3d, 0, 0, f)

        # for i in video_pose_3d_kpts.values():
        #     ax.scatter(q[f][i][0], q[f][i][1], q[f][i][2])
        line0.set_data(np.array([data_3d[f][15][0], data_3d[f][16][0]]), np.array([data_3d[f][15][1], data_3d[f][16][1]]))
        line0.set_3d_properties(np.array([data_3d[f][15][2], data_3d[f][16][2]]))

        line1.set_data(np.array([data_3d[f][15][0], data_3d[f][14][0]]), np.array([data_3d[f][15][1], data_3d[f][14][1]]))
        line1.set_3d_properties(np.array([data_3d[f][15][2], data_3d[f][14][2]]))

        line2.set_data(np.array([data_3d[f][13][0], data_3d[f][12][0]]), np.array([data_3d[f][13][1], data_3d[f][12][1]]))
        line2.set_3d_properties(np.array([data_3d[f][13][2], data_3d[f][12][2]]))

        line3.set_data(np.array([data_3d[f][11][0], data_3d[f][12][0]]), np.array([data_3d[f][11][1], data_3d[f][12][1]]))
        line3.set_3d_properties(np.array([data_3d[f][11][2], data_3d[f][12][2]]))

        line4.set_data(np.array([data_3d[f][11][0], data_3d[f][14][0]]), np.array([data_3d[f][11][1], data_3d[f][14][1]]))
        line4.set_3d_properties(np.array([data_3d[f][11][2], data_3d[f][14][2]]))

        line5.set_data(np.array([data_3d[f][11][0], data_3d[f][4][0]]), np.array([data_3d[f][11][1], data_3d[f][4][1]]))
        line5.set_3d_properties(np.array([data_3d[f][11][2], data_3d[f][4][2]]))

        line6.set_data(np.array([data_3d[f][1][0], data_3d[f][14][0]]), np.array([data_3d[f][1][1], data_3d[f][14][1]]))
        line6.set_3d_properties(np.array([data_3d[f][1][2], data_3d[f][14][2]]))

        line7.set_data(np.array([data_3d[f][1][0], data_3d[f][4][0]]), np.array([data_3d[f][1][1], data_3d[f][4][1]]))
        line7.set_3d_properties(np.array([data_3d[f][1][2], data_3d[f][4][2]]))

        line8.set_data(np.array([data_3d[f][1][0], data_3d[f][2][0]]), np.array([data_3d[f][1][1], data_3d[f][2][1]]))
        line8.set_3d_properties(np.array([data_3d[f][1][2], data_3d[f][2][2]]))

        line9.set_data(np.array([data_3d[f][2][0], data_3d[f][3][0]]), np.array([data_3d[f][2][1], data_3d[f][3][1]]))
        line9.set_3d_properties(np.array([data_3d[f][2][2], data_3d[f][3][2]]))

        line10.set_data(np.array([data_3d[f][5][0], data_3d[f][6][0]]), np.array([data_3d[f][5][1], data_3d[f][6][1]]))
        line10.set_3d_properties(np.array([data_3d[f][5][2], data_3d[f][6][2]]))

        line11.set_data(np.array([data_3d[f][4][0], data_3d[f][5][0]]), np.array([data_3d[f][4][1], data_3d[f][5][1]]))
        line11.set_3d_properties(np.array([data_3d[f][4][2], data_3d[f][5][2]]))

        plt.draw()
        plt.pause(0.1)


def rotate(data_3d, rotate_x=0, rotate_y=0, rotate_z=0):
    rotation_radians_x = np.radians(rotate_x)
    rotation_radians_y = np.radians(rotate_y)
    rotation_radians_z = np.radians(rotate_z)

    if rotate_x != 0:
        rotation_vector = rotation_radians_x * np.array([0, 0, 1])
        rotation = scipy.spatial.transform.Rotation.from_rotvec(rotation_vector)
        data_3d = np.array([np.array([rotation.apply(vec) for vec in fr]) for fr in data_3d])

    if rotate_y != 0:
        rotation_vector = rotation_radians_y * np.array([0, 1, 0])
        rotation = scipy.spatial.transform.Rotation.from_rotvec(rotation_vector)
        data_3d = np.array([np.array([rotation.apply(vec) for vec in fr]) for fr in data_3d])

    if rotate_z != 0:
        rotation_vector = rotation_radians_z * np.array([1, 0, 0])
        rotation = scipy.spatial.transform.Rotation.from_rotvec(rotation_vector)
        data_3d = np.array([np.array([rotation.apply(vec) for vec in fr]) for fr in data_3d])

    return data_3d
