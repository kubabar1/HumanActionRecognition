from har.impl.lstm_simple.evaluate import evaluate_tests, load_model
from har.utils.dataset_util import SetType, berkeley_mhad_classes, video_pose_3d_kpts, get_berkeley_dataset


def main():
    test_data, test_labels = get_berkeley_dataset('datasets_processed/berkeley/3D', set_type=SetType.TEST)
    model_path = 'results/lstm_simple_ep_10000_b_128_h_128_lr_0.0001_opt_RMSPROP_inp_STEP_mm_0.9_wd_0_hl_3_dr_0.5.pth'
    lstm_simple_model = load_model(model_path, len(berkeley_mhad_classes))

    accuracy = evaluate_tests(berkeley_mhad_classes, test_data, test_labels, lstm_simple_model, video_pose_3d_kpts)

    print('Test accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    main()