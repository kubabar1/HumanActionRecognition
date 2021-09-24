from har.impl.st_lstm.evaluate import load_model, evaluate_tests
from har.utils.dataset_util import SetType, berkeley_mhad_classes, video_pose_3d_kpts, get_berkeley_dataset


def main():
    test_data, test_labels = get_berkeley_dataset('datasets_processed/berkeley/3D', set_type=SetType.TEST)
    model_path = 'results/model_st_lstm_en_500_bs_128_lr_0.0001_op_RMSPROP_hs_128_it_SPLIT_dropout_0.5_momentum_0.9_wd_0_split_20_steps_32_lbd_0.5_bias_tau_2layers_2D.pth'
    p_simple_model = load_model(model_path, len(berkeley_mhad_classes))
    accuracy = evaluate_tests(berkeley_mhad_classes, test_data, test_labels, p_simple_model, video_pose_3d_kpts)
    print('Test accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    main()

# python draw_chart.py \
# --train-path /home/kuba/workspace/human_action_recognition/research/p_lstm/berkeley/20k/p_lstm_ntu_ep_10000_b_128_h_128_lr_0.0001_opt_RMSPROP_inp_SPLIT_mm_0.9_wd_0_hl_None_dr_0.5_split_20_steps_32_train_acc.npy \
# --validation-path /home/kuba/workspace/human_action_recognition/research/p_lstm/berkeley/20k/p_lstm_ntu_ep_10000_b_128_h_128_lr_0.0001_opt_RMSPROP_inp_SPLIT_mm_0.9_wd_0_hl_None_dr_0.5_split_20_steps_32_val_acc.npy \
# --epoch-count 10000 \
# --step 50
