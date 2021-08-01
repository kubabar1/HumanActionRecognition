from har.impl.p_lstm_ntu.batch_generator import get_batch_ntu_rgbd, get_batch_berkeley_mhad
from har.impl.p_lstm_ntu.train import train
# from har.impl.hierarchical_rnn.batch_generator import get_batch_ntu_rgbd, get_batch_berkeley_mhad
# from har.impl.hierarchical_rnn.train import train
# from har.impl.lstm_simple.batch_generator import get_batch_ntu_rgbd, get_batch_berkeley_mhad
# from har.impl.lstm_simple.train import train
from har.utils.dataset_utils import ntu_rgbd_classes, berkeley_mhad_classes


def main():
    train(berkeley_mhad_classes, get_batch_berkeley_mhad, dataset_path='./datasets/BerkeleyMHAD/3D', save_model=False,
          save_loss=True)


if __name__ == '__main__':
    main()
