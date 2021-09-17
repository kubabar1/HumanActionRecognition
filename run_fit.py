from random import randrange

from har.impl.st_lstm.evaluate import fit
from har.utils.dataset_util import get_berkeley_dataset, SetType, berkeley_mhad_classes, video_pose_3d_kpts, DatasetInputType


def main():
    test_data, test_labels = get_berkeley_dataset('datasets/BerkeleyMHAD/3D', set_type=SetType.TEST)
    random_id = randrange(len(test_labels))
    test_sequence, test_label = test_data[random_id], test_labels[random_id]
    model_path = 'results' \
                 '/st_lstm_ep_1000_b_128_h_128_lr_0.0001_opt_RMSPROP_inp_SPLIT_mm_0.9_wd_0_hl_None_dr_0.5_split_20_steps_32.pth'

    predicted = fit(berkeley_mhad_classes, test_sequence, model_path, video_pose_3d_kpts)

    print('CORRECT: {}'.format(berkeley_mhad_classes[test_label]))
    print('PREDICTED: {}'.format(predicted))


if __name__ == '__main__':
    main()
