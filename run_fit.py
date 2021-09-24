from random import randrange

from har.impl.st_lstm.evaluate import load_model, fit
from har.utils.dataset_util import get_berkeley_dataset, SetType, berkeley_mhad_classes, video_pose_3d_kpts


def main():
    test_data, test_labels = get_berkeley_dataset('datasets/BerkeleyMHAD/3D', set_type=SetType.TEST)
    random_id = randrange(len(test_labels))
    test_sequence, test_label = test_data[random_id], test_labels[random_id]
    model_path = 'results/model_st_lstm_en_500_bs_128_lr_0.0001_op_RMSPROP_hs_128_it_SPLIT_dropout_0.5_momentum_0.9_wd_0_split_20_steps_32_lbd_0.5_bias_tau_2layers_2D.pth'

    lstm_simple_model = load_model(model_path, len(berkeley_mhad_classes))

    predicted = fit(berkeley_mhad_classes, test_sequence, lstm_simple_model, video_pose_3d_kpts)

    print('CORRECT: {}'.format(berkeley_mhad_classes[test_label]))
    print('PREDICTED: {}'.format(predicted))


if __name__ == '__main__':
    main()
