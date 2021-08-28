from har.impl.lstm_simple.evaluate import evaluate_tests
from har.utils.dataset_util import get_berkeley_dataset_3d, SetType, berkeley_mhad_classes, get_ntu_rgbd_dataset_3d, \
    ntu_rgbd_classes, video_pose_3d_kpts


def main():
    test_data, test_labels = get_berkeley_dataset_3d('datasets_processed/berkeley/3D', set_type=SetType.TEST)
    model_path = 'results/lstm_simple_ep_20000_b_128_h_128_lr_0.0001_RMSPROP.pth'

    accuracy = evaluate_tests(berkeley_mhad_classes, test_data, test_labels, model_path, video_pose_3d_kpts)

    print('Test accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    main()
