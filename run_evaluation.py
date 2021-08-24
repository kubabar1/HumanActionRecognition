from har.impl.lstm_simple.evaluate import evaluate_tests
from har.utils.dataset_util import get_berkeley_dataset_3d, SetType, berkeley_mhad_classes, get_ntu_rgbd_dataset_3d, \
    ntu_rgbd_classes


def main():
    test_data, test_labels = get_berkeley_dataset_3d('datasets/BerkeleyMHAD/3D', set_type=SetType.TEST)
    model_path = 'results/lstm_simple_ep_10000_b_128_h_128_lr_1e-05_RMSPROP.pth'

    evaluate_tests(berkeley_mhad_classes, test_data, test_labels, model_path)


if __name__ == '__main__':
    main()
