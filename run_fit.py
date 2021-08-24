from random import randrange

from har.impl.lstm_simple.evaluate import evaluate_tests, fit
from har.utils.dataset_util import get_berkeley_dataset_3d, SetType, berkeley_mhad_classes, get_ntu_rgbd_dataset_3d, \
    ntu_rgbd_classes


def main():
    test_data, test_labels = get_berkeley_dataset_3d('datasets/BerkeleyMHAD/3D', set_type=SetType.TEST)
    random_id = randrange(len(test_labels))
    test_sequence, test_label = test_data[random_id], test_labels[random_id]
    model_path = 'results/lstm_simple_ep_10000_b_128_h_128_lr_1e-05_RMSPROP.pth'

    predicted = fit(berkeley_mhad_classes, test_sequence, model_path)

    print('CORRECT: {}'.format(berkeley_mhad_classes[test_label]))
    print('PREDICTED: {}'.format(predicted))


if __name__ == '__main__':
    main()
