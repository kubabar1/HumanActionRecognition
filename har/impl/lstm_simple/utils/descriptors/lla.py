import math

import numpy as np


def calculate_lla(keypoints_sequence, analysed_lines):
    lla_batch = []
    for frame in keypoints_sequence[:, :, :]:
        frame_tmp = []
        for line_i_id, line_i in enumerate(analysed_lines):
            for line_j_id, line_j in enumerate(analysed_lines):
                if line_i_id != line_j_id:
                    joint_j = frame[line_i[0]]
                    joint_k = frame[line_i[1]]
                    jjo_1 = (joint_j - joint_k) / np.linalg.norm(joint_j - joint_k)
                    joint_j2 = frame[line_j[0]]
                    joint_k2 = frame[line_j[1]]
                    jjo_2 = (joint_j2 - joint_k2) / np.linalg.norm(joint_j2 - joint_k2)
                    tmp = np.matmul(jjo_1, jjo_2)
                    if tmp > 1:
                        tmp = 1
                    elif tmp < -1:
                        tmp = -1
                    res = math.acos(tmp)
                    frame_tmp.append(res)
        lla_batch.append(np.array(frame_tmp))
    return np.array(lla_batch)
