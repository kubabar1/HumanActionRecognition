# from har.impl.p_lstm_ntu.evaluate import evaluate_tests
# from har.impl.st_lstm.evaluate import evaluate_tests
from har.impl.jtm.evaluate import evaluate_tests, ModelType
from har.utils.dataset_util import SetType, berkeley_mhad_classes, video_pose_3d_kpts, get_berkeley_dataset, berkeley_frame_height, \
    berkeley_frame_width


def main():
    test_data, test_labels = get_berkeley_dataset('datasets_processed/berkeley/3D', set_type=SetType.TEST)
    model_path = './results' \
                 '/jtm_ep_20_b_32_h_None_lr_0.0001_opt_SGD_inp_None_mm_None_wd_None_hl_None_dr_None_split_None_steps_None_front.pth'

    accuracy = evaluate_tests(berkeley_mhad_classes, test_data, test_labels, model_path, video_pose_3d_kpts, berkeley_frame_width,
                              berkeley_frame_height, ModelType.FRONT)

    print('Test accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    main()

# python draw_chart.py \
# --train-path /home/kuba/workspace/human_action_recognition/research/p_lstm/berkeley/20k/p_lstm_ntu_ep_10000_b_128_h_128_lr_0.0001_opt_RMSPROP_inp_SPLIT_mm_0.9_wd_0_hl_None_dr_0.5_split_20_steps_32_train_acc.npy \
# --validation-path /home/kuba/workspace/human_action_recognition/research/p_lstm/berkeley/20k/p_lstm_ntu_ep_10000_b_128_h_128_lr_0.0001_opt_RMSPROP_inp_SPLIT_mm_0.9_wd_0_hl_None_dr_0.5_split_20_steps_32_val_acc.npy \
# --epoch-count 10000 \
# --step 50
