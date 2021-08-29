import argparse

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


def draw_chart(train_path_list, validation_path_list, epoch_count, step_size=25, use_interpolation=True, hide_train=False,
               hide_validation=False, legends_arr=None, save_results=False, results_path='result_chart.png'):
    if legends_arr is not None:
        for train_path_single, validation_path_single, legends_single in zip(train_path_list, validation_path_list, legends_arr):
            single_plot(train_path_single, validation_path_single, epoch_count, 'trening_' + legends_single, 'walidacja_' + legends_single,
                        step_size, use_interpolation, hide_train, hide_validation, save_results, results_path)
    else:
        for train_path_single, validation_path_single in zip(train_path_list, validation_path_list):
            single_plot(train_path_single, validation_path_single, epoch_count, 'trening', 'walidacja', step_size, use_interpolation,
                        hide_train, hide_validation, save_results, results_path)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5, prop={"size": 18})
    plt.show()


def single_plot(train_path, validation_path, epoch_count, train_legend='', validation_legend='', step=25, use_interpolation=True,
                hide_train=False, hide_validation=False, save_results=False, results_path='result_chart.png'):
    train = np.load(train_path)
    validation = np.load(validation_path)

    iterations_count = len(train)
    el = int(iterations_count / step)
    validation_st = int(iterations_count / (len(validation) + 1))

    if len(train) != len(validation):
        if use_interpolation:
            x_interp = np.arange(0, iterations_count - validation_st, validation_st)
            interp = interp1d(x_interp, validation)
            validation = interp(np.arange(0, x_interp[-1]))
        else:
            validation = [validation[int(i / validation_st)] for i in range(iterations_count - validation_st)]

    train = train[:len(validation)]

    train = [np.mean(train[i * step:i * step + step]) for i in range(el)]
    validation = [np.mean(validation[i * step:i * step + step]) for i in range(el)]

    cut = (iterations_count - int(iterations_count / step) * step)

    ep = int(epoch_count / (iterations_count - cut + step))

    if ep <= 0:
        ep = 1

    if not hide_train:
        plt.plot(np.arange(0, iterations_count - cut, step) * ep, train, label=train_legend)
    if not hide_validation:
        plt.plot(np.arange(0, iterations_count - cut, step) * ep, validation, label=validation_legend)
    if save_results:
        plt.savefig(results_path)


def validate_input(train_path_list_args_val, validation_path_list_args_val, legends_list_args_val):
    if legends_list_args is not None:
        if len(legends_list_args_val) != len(train_path_list_args_val):
            raise Exception('Legends array length must be equal to training and validation input path data length')

    if len(train_path_list_args_val) != len(validation_path_list_args_val):
        raise Exception('Training data length must be equal to validation input path data length')


if __name__ == '__main__':
    # train_path = '../results/lstm_simple_ep_10000_b_128_h_128_lr_1e-05_RMSPROP_train_acc.npy'
    # validation_path = '../results/lstm_simple_ep_10000_b_128_h_128_lr_1e-05_RMSPROP_val_acc.npy'
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', help='Absolute path to *.npy file with train results', required=True, action='append')
    parser.add_argument('--validation-path', help='Absolute path to *.npy file with validation results', required=True, action='append')
    parser.add_argument('--epoch-count', help='Count of epochs used during training', required=True)
    parser.add_argument('--legends', help='Name for legends', required=False, action='append', default=None)
    parser.add_argument('--results-path', help='Path where generated draw will be saved', required=False,
                        default='result_chart.png')
    parser.add_argument('--step', help='Step size', default=25)
    parser.add_argument('--hide-train', help='Hide train plot', default=False, action='store_true')
    parser.add_argument('--hide-validation', help='Hide validation plot', default=False, action='store_true')
    parser.add_argument('--save-results', help='Save result chart', default=False, action='store_true')

    args = parser.parse_args()
    train_path_list_args = args.train_path
    validation_path_list_args = args.validation_path
    legends_list_args = args.legends
    step_size_args = int(args.step)
    hide_train_arg = args.hide_train
    hide_validation_arg = args.hide_validation
    save_results_args = args.save_results
    results_path_args = args.results_path
    epoch_count_args = int(args.epoch_count)

    validate_input(train_path_list_args, validation_path_list_args, legends_list_args)

    draw_chart(train_path_list_args, validation_path_list_args, step_size=step_size_args, hide_train=hide_train_arg,
               hide_validation=hide_validation_arg, legends_arr=legends_list_args, save_results=save_results_args,
               results_path=results_path_args, epoch_count=epoch_count_args)
