# from har.impl.jtm.train import train
from har.impl.lstm_simple.train import train
# from har.impl.geometric_multilayer_lstm.train import train
# from har.impl.trust_gates_st_lstm.train import train
# from har.impl.hierarchical_rnn.train import train
from har.utils.dataset_util import get_berkeley_dataset_3d, SetType, berkeley_mhad_classes, get_ntu_rgbd_dataset_3d, \
    ntu_rgbd_classes, video_pose_3d_kpts, DatasetInputType, utd_mhad_classes, get_utd_mhad_dataset_3d


def main():
    # training_data, training_labels = get_berkeley_dataset_3d('datasets_processed/berkeley/3D', set_type=SetType.TRAINING)
    # validation_data, validation_labels = get_berkeley_dataset_3d('datasets_processed/berkeley/3D', set_type=SetType.VALIDATION)
    # training_data, training_labels = get_ntu_rgbd_dataset_3d('datasets_processed/nturgbd/3D', set_type=SetType.TRAINING)
    # validation_data, validation_labels = get_ntu_rgbd_dataset_3d('datasets_processed/nturgbd/3D', set_type=SetType.VALIDATION)
    training_data, training_labels = get_utd_mhad_dataset_3d('datasets_processed/utd_mhad/3D', set_type=SetType.TRAINING)
    validation_data, validation_labels = get_utd_mhad_dataset_3d('datasets_processed/utd_mhad/3D', set_type=SetType.VALIDATION)

    train(utd_mhad_classes, training_data, training_labels, validation_data, validation_labels, video_pose_3d_kpts)


if __name__ == '__main__':
    main()
