from har.impl.lstm_simple.train import train
from har.utils.batch_generator import get_batch_ntu_rgbd
from har.utils.dataset_utils import ntu_rgbd_classes


def main():
    train(ntu_rgbd_classes, get_batch_ntu_rgbd, dataset_path='./datasets/nturgbd/3D')


if __name__ == '__main__':
    main()
