import math
import numpy as np
from scipy.spatial import distance


def heron_formula(kpt_a, kpt_b, kpt_c):
    a = distance.euclidean(kpt_a, kpt_b)
    b = distance.euclidean(kpt_b, kpt_c)
    c = distance.euclidean(kpt_c, kpt_a)
    s = (a + b + c) / 2
    return math.sqrt(s * (s - a) * (s - b) * (s - c))


def create_jld_batch(keypoints_sequence, analysed_lines, analysed_kpts):
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


def batch_test():
    kpts_sequence = np.random.rand(150, 17, 3)
    analysed_kpts = [16, 15, 14, 11, 12, 13, 1, 2, 3, 4, 5, 6]
    analysed_lines_1 = [[3, 2], [2, 1], [16, 15], [15, 14], [12, 11], [13, 12], [5, 4], [6, 5]]
    analysed_lines_2 = [[3, 1], [16, 14], [13, 11], [6, 4]]
    analysed_lines_3 = [[3, 16], [6, 16], [3, 6], [16, 6], [13, 6], [16, 13]]
    analysed_lines = analysed_lines_1 + analysed_lines_2 + analysed_lines_3
    jld_batch_tmp = create_jld_batch(kpts_sequence, analysed_lines, analysed_kpts)
    print(jld_batch_tmp.shape)

# batch_test()
