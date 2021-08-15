from har.impl.jtm.train import train
# from har.impl.lstm_simple.train import train
# from har.impl.geometric_multilayer_lstm.train import train
from har.utils.dataset_util import get_berkeley_dataset_3d, SetType, berkeley_mhad_classes, berkeley_frame_height, \
    berkeley_frame_width


def main():
    training_data, training_labels = get_berkeley_dataset_3d('datasets/BerkeleyMHAD/3D', set_type=SetType.TRAINING)
    validation_data, validation_labels = get_berkeley_dataset_3d('datasets/BerkeleyMHAD/3D', set_type=SetType.VALIDATION)

    train(berkeley_mhad_classes, training_data, training_labels, validation_data, validation_labels, berkeley_frame_width,
          berkeley_frame_height)


if __name__ == '__main__':
    main()
