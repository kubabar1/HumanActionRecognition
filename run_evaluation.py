# from har.impl.p_lstm_ntu.evaluate import evaluate_tests
from har.impl.st_lstm.evaluate import evaluate_tests
from har.utils.dataset_util import SetType, berkeley_mhad_classes, video_pose_3d_kpts, get_berkeley_dataset


def main():
    test_data, test_labels = get_berkeley_dataset('datasets_processed/berkeley/3D', set_type=SetType.TEST)
    model_path = './results' \
                 '/st_lstm_ep_1000_b_128_h_128_lr_0.0001_opt_RMSPROP_inp_SPLIT_mm_0.9_wd_0_hl_None_dr_0.5_split_20_steps_32.pth'

    accuracy = evaluate_tests(berkeley_mhad_classes, test_data, test_labels, model_path, video_pose_3d_kpts)

    print('Test accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    main()

# python draw_chart.py \
# --train-path /home/kuba/workspace/human_action_recognition/research/p_lstm/berkeley/20k/p_lstm_ntu_ep_10000_b_128_h_128_lr_0.0001_opt_RMSPROP_inp_SPLIT_mm_0.9_wd_0_hl_None_dr_0.5_split_20_steps_32_train_acc.npy \
# --validation-path /home/kuba/workspace/human_action_recognition/research/p_lstm/berkeley/20k/p_lstm_ntu_ep_10000_b_128_h_128_lr_0.0001_opt_RMSPROP_inp_SPLIT_mm_0.9_wd_0_hl_None_dr_0.5_split_20_steps_32_val_acc.npy \
# --epoch-count 10000 \
# --step 50
