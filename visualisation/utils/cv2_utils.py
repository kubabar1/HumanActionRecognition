import math as m
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
                                'Mean accuracy: ' + str(round(np.sum([i[2] for i in pose_kpts]) / 17, 3)),
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


close = False


def close_window(event):
    global close
    close = True


from math import *


def angle_trunc(a):
    while a < 0.0:
        a += pi * 2
    return a


def getAngleBetweenPoints(x_orig, y_orig, x_landmark, y_landmark):
    deltaY = y_landmark - y_orig
    deltaX = x_landmark - x_orig
    return angle_trunc(atan2(deltaY, deltaX))


def draw_3d_pose(data_3d, kpts_description, pause=0.1):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    right_wrist = kpts_description['right_wrist']
    left_wrist = kpts_description['left_wrist']
    right_elbow = kpts_description['right_elbow']
    left_elbow = kpts_description['left_elbow']
    right_shoulder = kpts_description['right_shoulder']
    left_shoulder = kpts_description['left_shoulder']
    right_hip = kpts_description['right_hip']
    left_hip = kpts_description['left_hip']
    right_knee = kpts_description['right_knee']
    left_knee = kpts_description['left_knee']
    right_ankle = kpts_description['right_ankle']
    left_ankle = kpts_description['left_ankle']

    data_3d = normalise_skeleton(data_3d, left_hip, right_hip)
    # print(len(data_3d[0]))
    # print(data_3d.shape)
    data_3d = np.array([[[-k[0], -k[1], k[2]] for k in f] for f in data_3d])

    line0, = ax.plot(np.array([data_3d[0][right_elbow][0], data_3d[0][right_wrist][0]]),
                     np.array([data_3d[0][right_elbow][1], data_3d[0][right_wrist][1]]),
                     np.array([data_3d[0][right_elbow][2], data_3d[0][right_wrist][2]]))
    line1, = ax.plot(np.array([data_3d[0][right_elbow][0], data_3d[0][right_shoulder][0]]),
                     np.array([data_3d[0][right_elbow][1], data_3d[0][right_shoulder][1]]),
                     np.array([data_3d[0][right_elbow][2], data_3d[0][right_shoulder][2]]))

    line2, = ax.plot(np.array([data_3d[0][left_wrist][0], data_3d[0][left_elbow][0]]),
                     np.array([data_3d[0][left_wrist][1], data_3d[0][left_elbow][1]]),
                     np.array([data_3d[0][left_wrist][2], data_3d[0][left_elbow][2]]))
    line3, = ax.plot(np.array([data_3d[0][left_shoulder][0], data_3d[0][left_elbow][0]]),
                     np.array([data_3d[0][left_shoulder][1], data_3d[0][left_elbow][1]]),
                     np.array([data_3d[0][left_shoulder][2], data_3d[0][left_elbow][2]]))

    line4, = ax.plot(np.array([data_3d[0][left_shoulder][0], data_3d[0][right_shoulder][0]]),
                     np.array([data_3d[0][left_shoulder][1], data_3d[0][right_shoulder][1]]),
                     np.array([data_3d[0][left_shoulder][2], data_3d[0][right_shoulder][2]]))

    line5, = ax.plot(np.array([data_3d[0][left_shoulder][0], data_3d[0][left_hip][0]]),
                     np.array([data_3d[0][left_shoulder][1], data_3d[0][left_hip][1]]),
                     np.array([data_3d[0][left_shoulder][2], data_3d[0][left_hip][2]]))
    line6, = ax.plot(np.array([data_3d[0][right_hip][0], data_3d[0][right_shoulder][0]]),
                     np.array([data_3d[0][right_hip][1], data_3d[0][right_shoulder][1]]),
                     np.array([data_3d[0][right_hip][2], data_3d[0][right_shoulder][2]]))

    line7, = ax.plot(np.array([data_3d[0][right_hip][0], data_3d[0][left_hip][0]]),
                     np.array([data_3d[0][right_hip][1], data_3d[0][left_hip][1]]),
                     np.array([data_3d[0][right_hip][2], data_3d[0][left_hip][2]]))

    line8, = ax.plot(np.array([data_3d[0][right_hip][0], data_3d[0][right_knee][0]]),
                     np.array([data_3d[0][right_hip][1], data_3d[0][right_knee][1]]),
                     np.array([data_3d[0][right_hip][2], data_3d[0][right_knee][2]]))
    line9, = ax.plot(np.array([data_3d[0][right_knee][0], data_3d[0][right_ankle][0]]),
                     np.array([data_3d[0][right_knee][1], data_3d[0][right_ankle][1]]),
                     np.array([data_3d[0][right_knee][2], data_3d[0][right_ankle][2]]))

    line10, = ax.plot(np.array([data_3d[0][left_knee][0], data_3d[0][left_ankle][0]]),
                      np.array([data_3d[0][left_knee][1], data_3d[0][left_ankle][1]]),
                      np.array([data_3d[0][left_knee][2], data_3d[0][left_ankle][2]]))
    line11, = ax.plot(np.array([data_3d[0][left_knee][0], data_3d[0][left_hip][0]]),
                      np.array([data_3d[0][left_knee][1], data_3d[0][left_hip][1]]),
                      np.array([data_3d[0][left_knee][2], data_3d[0][left_hip][2]]))

    # ax.view_init(90, 90)
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    axs = fig.gca(projection='3d')

    xLabel = axs.set_xlabel('x')
    yLabel = axs.set_ylabel('y')
    zLabel = axs.set_zlabel('z')

    # print(data_3d[0][right_hip][2])
    #
    # rot = atan((data_3d[0][right_hip][2] - data_3d[0][left_hip][2]) / (data_3d[0][right_hip][0] - data_3d[0][left_hip][0]))
    # print(rot)
    #
    # ry = Ry(rot)
    #
    # data_3d = np.array([[ry * np.array([[k[0]], [k[1]], [k[2]]]) for k in f] for f in data_3d])
    #
    # # data_3d = rotate(data_3d, 0, angleInDegrees, 0)
    #
    # rot = atan((data_3d[0][right_hip][2] - data_3d[0][left_hip][2]) / (data_3d[0][right_hip][0] - data_3d[0][left_hip][0]))
    # print(rot)
    # data_3d = data_3d.reshape((-1, 17, 3))

    for f in range(1, len(data_3d)):
        data = data_3d[f]

        # for i in video_pose_3d_kpts.values():
        #     ax.scatter(q[f][i][0], q[f][i][1], q[f][i][2])
        line0.set_data(np.array([data[right_elbow][0], data[right_wrist][0]]),
                       np.array([data[right_elbow][1], data[right_wrist][1]]))
        line0.set_3d_properties(np.array([data[right_elbow][2], data[right_wrist][2]]))

        line1.set_data(np.array([data[right_elbow][0], data[right_shoulder][0]]),
                       np.array([data[right_elbow][1], data[right_shoulder][1]]))
        line1.set_3d_properties(np.array([data[right_elbow][2], data[right_shoulder][2]]))

        line2.set_data(np.array([data[left_wrist][0], data[left_elbow][0]]),
                       np.array([data[left_wrist][1], data[left_elbow][1]]))
        line2.set_3d_properties(np.array([data[left_wrist][2], data[left_elbow][2]]))

        line3.set_data(np.array([data[left_shoulder][0], data[left_elbow][0]]),
                       np.array([data[left_shoulder][1], data[left_elbow][1]]))
        line3.set_3d_properties(np.array([data[left_shoulder][2], data[left_elbow][2]]))

        line4.set_data(np.array([data[left_shoulder][0], data[right_shoulder][0]]),
                       np.array([data[left_shoulder][1], data[right_shoulder][1]]))
        line4.set_3d_properties(np.array([data[left_shoulder][2], data[right_shoulder][2]]))

        line5.set_data(np.array([data[left_shoulder][0], data[left_hip][0]]),
                       np.array([data[left_shoulder][1], data[left_hip][1]]))
        line5.set_3d_properties(np.array([data[left_shoulder][2], data[left_hip][2]]))

        line6.set_data(np.array([data[right_hip][0], data[right_shoulder][0]]),
                       np.array([data[right_hip][1], data[right_shoulder][1]]))
        line6.set_3d_properties(np.array([data[right_hip][2], data[right_shoulder][2]]))

        line7.set_data(np.array([data[right_hip][0], data[left_hip][0]]),
                       np.array([data[right_hip][1], data[left_hip][1]]))
        line7.set_3d_properties(np.array([data[right_hip][2], data[left_hip][2]]))

        line8.set_data(np.array([data[right_hip][0], data[right_knee][0]]),
                       np.array([data[right_hip][1], data[right_knee][1]]))
        line8.set_3d_properties(np.array([data[right_hip][2], data[right_knee][2]]))

        line9.set_data(np.array([data[right_knee][0], data[right_ankle][0]]),
                       np.array([data[right_knee][1], data[right_ankle][1]]))
        line9.set_3d_properties(np.array([data[right_knee][2], data[right_ankle][2]]))

        line10.set_data(np.array([data[left_knee][0], data[left_ankle][0]]),
                        np.array([data[left_knee][1], data[left_ankle][1]]))
        line10.set_3d_properties(np.array([data[left_knee][2], data[left_ankle][2]]))

        line11.set_data(np.array([data[left_hip][0], data[left_knee][0]]),
                        np.array([data[left_hip][1], data[left_knee][1]]))
        line11.set_3d_properties(np.array([data[left_hip][2], data[left_knee][2]]))

        plt.draw()
        plt.pause(pause)
        fig.canvas.mpl_connect('close_event', close_window)
        if close:
            plt.close('all')
            break


def normalise_skeleton(data_3d, left_hip_index, right_hip_index):
    return scale_skeleton(move_hip_to_center(data_3d, left_hip_index, right_hip_index))


def scale_skeleton(data_3d):
    s = 1 / np.max(np.abs(data_3d))
    return data_3d * s


def move_hip_to_center(data_3d, left_hip_index, right_hip_index):
    dd = (data_3d[:, left_hip_index, :] + data_3d[:, right_hip_index, :]) / 2
    return np.array([[k - dd[i] for k in d] for i, d in enumerate(data_3d)])


def Ry(theta):
    return np.matrix([[m.cos(theta), 0, m.sin(theta)],
                      [0, 1, 0],
                      [-m.sin(theta), 0, m.cos(theta)]])


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
