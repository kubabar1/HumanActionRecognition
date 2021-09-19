import numpy as np


def calculate_rp(keypoints_sequence, analysed_kpts):
    rp_batch = []
    for frame in keypoints_sequence[:, analysed_kpts, :]:
        frame_tmp = []
        for joint_j_id, joint_j in enumerate(frame):
            for joint_k_id, joint_k in enumerate(frame):
                if joint_j_id != joint_k_id:
                    frame_tmp.append(joint_j - joint_k)
        rp_batch.append(np.array(frame_tmp))
    return np.array(rp_batch).reshape(len(keypoints_sequence), -1)
