# from har.impl.p_lstm_ntu.batch_generator import get_batch_ntu_rgbd, get_batch_berkeley_mhad
# from har.impl.p_lstm_ntu.train import train
# from har.impl.hierarchical_rnn.batch_generator import get_batch_ntu_rgbd, get_batch_berkeley_mhad
# from har.impl.hierarchical_rnn.train import train
# from har.impl.lstm_simple.train import train
from har.impl.jtm.train import train
# from har.utils.dataset_utils import ntu_rgbd_classes, berkeley_mhad_classes
from har.utils.dataset_util import get_berkeley_dataset_3d, SetType, berkeley_mhad_classes, berkeley_frame_width, \
    berkeley_frame_height, get_analysed_keypoints


def main():
    training_data, training_labels = get_berkeley_dataset_3d('datasets/BerkeleyMHAD/3D', set_type=SetType.TRAINING)
    validation_data, validation_labels = get_berkeley_dataset_3d('datasets/BerkeleyMHAD/3D', set_type=SetType.VALIDATION)


    train(berkeley_mhad_classes, training_data, training_labels, validation_data, validation_labels, berkeley_frame_width,
          berkeley_frame_height)

    # train(berkeley_mhad_classes, training_data, training_labels, validation_data, validation_labels)


if __name__ == '__main__':
    main()
