def calculate_jjc(keypoints_sequence, analysed_kpts):
    frames_count, kpts_count, cord_count = keypoints_sequence.shape
    return keypoints_sequence[:, analysed_kpts, :].reshape((frames_count, len(analysed_kpts) * cord_count))
