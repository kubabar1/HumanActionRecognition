import numpy as np
from scipy.spatial import distance


def calculate_jjd(keypoints_sequence, analysed_kpts):
    jjd_batch = []
    for frame in keypoints_sequence[:, analysed_kpts, :]:
        frame_tmp = []
        for joint_j_id, joint_j in enumerate(frame):
            for joint_k_id, joint_k in enumerate(frame):
                if joint_j_id != joint_k_id:
                    frame_tmp.append(distance.euclidean(joint_j, joint_k))
        jjd_batch.append(np.array(frame_tmp))
    return np.array(jjd_batch)
