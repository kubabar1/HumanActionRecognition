from random import randrange

from har.impl.jtm.evaluate import fit, ModelType
from har.utils.dataset_util import get_berkeley_dataset, SetType, berkeley_mhad_classes, video_pose_3d_kpts, berkeley_frame_width, \
    berkeley_frame_height


def main():
    test_data, test_labels = get_berkeley_dataset('datasets/BerkeleyMHAD/3D', set_type=SetType.TEST)
    random_id = randrange(len(test_labels))
    test_sequence, test_label = test_data[random_id], test_labels[random_id]
    model_path = './results' \
                 '/jtm_ep_20_b_32_h_None_lr_0.0001_opt_SGD_inp_None_mm_None_wd_None_hl_None_dr_None_split_None_steps_None_front.pth'

    predicted = fit(berkeley_mhad_classes, test_sequence, model_path, video_pose_3d_kpts, berkeley_frame_width, berkeley_frame_height,
                    ModelType.FRONT)

    print('CORRECT: {}'.format(berkeley_mhad_classes[test_label]))
    print('PREDICTED: {}'.format(predicted))


if __name__ == '__main__':
    main()
