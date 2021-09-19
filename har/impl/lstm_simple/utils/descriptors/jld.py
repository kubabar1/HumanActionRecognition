import math

import numpy as np
from scipy.spatial import distance


def heron_formula(kpt_a, kpt_b, kpt_c):
    a = distance.euclidean(kpt_a, kpt_b)
    b = distance.euclidean(kpt_b, kpt_c)
    c = distance.euclidean(kpt_c, kpt_a)
    s = (a + b + c) / 2
    tmp = s * (s - a) * (s - b) * (s - c)
    return math.sqrt(tmp if tmp > 0 else 0)


def calculate_jld(keypoints_sequence, analysed_kpts, analysed_lines):
    jld_batch = []
    for frame in keypoints_sequence:
        frame_tmp = []
        for line in analysed_lines:
            joint_j, joint_k = line
            for joint_n in analysed_kpts:
                if joint_n != joint_j and joint_n != joint_k:
                    joint_j_c = frame[joint_j]
                    joint_k_c = frame[joint_k]
                    joint_n_c = frame[joint_n]
                    jld = 2 * heron_formula(joint_j_c, joint_k_c, joint_n_c) / distance.euclidean(joint_j_c, joint_k_c)
                    frame_tmp.append(jld)
        jld_batch.append(np.array(frame_tmp))
    return np.array(jld_batch)
