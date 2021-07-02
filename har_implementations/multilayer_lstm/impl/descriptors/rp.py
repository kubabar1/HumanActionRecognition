import numpy as np


def create_rp_batch(keypoints_sequence, analysed_kpts):
    keypoints_sequence = keypoints_sequence[:, analysed_kpts, :]
    rp_batch = []
    for frame in keypoints_sequence:
        frame_tmp = []
        for joint_j_id, joint_j in enumerate(frame):
            for joint_k_id, joint_k in enumerate(frame):
                if joint_j_id != joint_k_id:
                    frame_tmp.append(joint_j - joint_k)
        rp_batch.append(np.array(frame_tmp))
    return np.array(rp_batch).reshape(len(keypoints_sequence), -1)


def batch_test():
    kpts_sequence = np.random.rand(150, 17, 3)
    analysed_kpts = [16, 15, 14, 11, 12, 13, 1, 2, 3, 4, 5, 6]
    rp_batch_tmp = create_rp_batch(kpts_sequence, analysed_kpts)
    print(rp_batch_tmp.shape)

# batch_test()
