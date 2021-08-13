# from har.impl.p_lstm_ntu.batch_generator import get_batch_ntu_rgbd, get_batch_berkeley_mhad
# from har.impl.p_lstm_ntu.train import train
# from har.impl.hierarchical_rnn.batch_generator import get_batch_ntu_rgbd, get_batch_berkeley_mhad
# from har.impl.hierarchical_rnn.train import train
# from har.impl.lstm_simple.batch_generator import get_batch_ntu_rgbd, get_batch_berkeley_mhad
# from har.impl.lstm_simple.train import train
# from har.utils.dataset_utils import ntu_rgbd_classes, berkeley_mhad_classes
from dataset_util import get_berkeley_dataset_3d, SetType, berkeley_mhad_classes, berkeley_frame_width, berkeley_frame_height, \
    get_analysed_keypoints
from har.impl.jtm.train import train


def main():
    # train(ntu_rgbd_classes, get_batch_ntu_rgbd, dataset_path='./datasets/nturgbd/3D', save_model=False, save_loss=True)
    # train(berkeley_mhad_classes, get_batch_berkeley_mhad, dataset_path='./datasets/BerkeleyMHAD/3D', save_model=True,
    #       save_loss=True)

    training_data, training_labels = get_berkeley_dataset_3d('datasets/BerkeleyMHAD/3D', set_type=SetType.TRAINING)
    validation_data, validation_labels = get_berkeley_dataset_3d('datasets/BerkeleyMHAD/3D', set_type=SetType.VALIDATION)

    analysed_kpts_left, analysed_kpts_right = get_analysed_keypoints()

    train(berkeley_mhad_classes, training_data, training_labels, validation_data, validation_labels, berkeley_frame_width,
          berkeley_frame_height, analysed_kpts_left, analysed_kpts_right)


if __name__ == '__main__':
    main()
