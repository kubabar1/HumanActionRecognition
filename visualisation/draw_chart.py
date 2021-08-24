import argparse

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


def draw_chart(train_path, test_path, step=25, use_interpolation=True):
    train = np.load(train_path)
    test = np.load(test_path)

    iterations_count = len(train)
    el = int(iterations_count / step)
    test_st = int(iterations_count / (len(test) + 1))

    if len(train) != len(test):
        if use_interpolation:
            x_interp = np.arange(0, iterations_count - test_st, test_st)
            interp = interp1d(x_interp, test)
            test = interp(np.arange(0, x_interp[-1]))
        else:
            test = [test[int(i / test_st)] for i in range(iterations_count - test_st)]

    train = train[:len(test)]

    train = [np.mean(train[i * step:i * step + step]) for i in range(el)]
    test = [np.mean(test[i * step:i * step + step]) for i in range(el)]

    cut = (iterations_count - int(iterations_count / step) * step)

    plt.plot(np.arange(0, iterations_count - cut, step), train, label='train')
    plt.plot(np.arange(0, iterations_count - cut, step), test, label='test')

    plt.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    # train_path = '../results/lstm_simple_ep_10000_b_128_h_128_lr_1e-05_RMSPROP_train_acc.npy'
    # test_path = '../results/lstm_simple_ep_10000_b_128_h_128_lr_1e-05_RMSPROP_val_acc.npy'
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', help='Absolute path to *.npy file with train results', required=True)
    parser.add_argument('--test-path', help='Absolute path to *.npy file with test results', required=True)
    parser.add_argument('--step', help='Step size', default=25)

    args = parser.parse_args()
    train_path = args.train_path
    test_path = args.test_path
    step = int(args.step)

    draw_chart(train_path, test_path, step=step)
