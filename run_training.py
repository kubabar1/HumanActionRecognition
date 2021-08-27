# from har.impl.jtm.train import train
from har.impl.lstm_simple.train import train
# from har.impl.geometric_multilayer_lstm.train import train
# from har.impl.trust_gates_st_lstm.train import train
from har.utils.dataset_util import get_berkeley_dataset_3d, SetType, berkeley_mhad_classes, get_ntu_rgbd_dataset_3d, \
    ntu_rgbd_classes, video_pose_3d_kpts


def main():
    training_data, training_labels = get_berkeley_dataset_3d('datasets_processed/berkeley/3D', set_type=SetType.TRAINING)
    validation_data, validation_labels = get_berkeley_dataset_3d('datasets_processed/berkeley/3D', set_type=SetType.VALIDATION)
    # training_data, training_labels = get_ntu_rgbd_dataset_3d('datasets/nturgbd/3D', set_type=SetType.TRAINING)
    # validation_data, validation_labels = get_ntu_rgbd_dataset_3d('datasets/nturgbd/3D', set_type=SetType.VALIDATION)

    train(berkeley_mhad_classes, training_data, training_labels, validation_data, validation_labels, video_pose_3d_kpts,
          epoch_nb=10000)


if __name__ == '__main__':
    main()
