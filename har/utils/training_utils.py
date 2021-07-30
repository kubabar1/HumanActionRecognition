import math
import os
import pathlib
import time
from enum import Enum, auto

import numpy as np
import torch
from matplotlib import pyplot as plt


class Optimizer(Enum):
    RMSPROP = auto()
    SGD = auto()
    ADAM = auto()


def test_model(model, criterion, classes, get_batch, dataset_path, batch_size, device, epoch, epoch_nb, print_test_every,
               all_test_losses, start_time):
    with torch.no_grad():
        data_test, test_y = get_batch(dataset_path, batch_size, is_training=False)
        tensor_test_y = torch.from_numpy(test_y).to(device)
        tensor_test_x = torch.tensor(data_test.reshape((data_test.shape[0], data_test.shape[1], -1)), dtype=torch.float,
                                     device=device)
        output_test = model(tensor_test_x)
        loss_test = criterion(output_test, tensor_test_y)

        ctgs_test = [classes[int(tty)] for tty in tensor_test_y]
        gss_test = [classes[int(torch.argmax(torch.exp(o)).item())] for o in output_test]
        correct_pred_in_batch_test = len([1 for c, g in zip(ctgs_test, gss_test) if c == g])

        category_test = classes[int(tensor_test_y[0])]
        guess_test = classes[int(torch.argmax(torch.exp(output_test)[0]).item())]

        correct_test = '✓' if guess_test == category_test else '✗ (%s)' % category_test

        if epoch % print_test_every == 0:
            print('TEST: %d %d%% (%s) %.4f  / %s %s [%d/%d -> %.2f%%]' % (
                epoch, epoch / epoch_nb * 100, time_since(start_time), loss_test, guess_test, correct_test,
                correct_pred_in_batch_test, batch_size, correct_pred_in_batch_test / batch_size * 100))
        all_test_losses.append(loss_test.cpu().detach())


def print_train_results(classes, output, tensor_train_y, epoch, epoch_nb, start_time, loss, batch_size):
    ctgs = [classes[int(tty)] for tty in tensor_train_y]
    gss = [classes[int(torch.argmax(torch.exp(o)).item())] for o in output]
    correct_pred_in_batch = len([1 for c, g in zip(ctgs, gss) if c == g])

    category = classes[int(tensor_train_y[0])]
    guess = classes[int(torch.argmax(torch.exp(output)[0]).item())]

    correct = '✓' if guess == category else '✗ (%s)' % category

    print('TRAIN: %d %d%% (%s) %.4f  / %s %s [%d/%d -> %.2f%%]' % (
        epoch, epoch / epoch_nb * 100, time_since(start_time), loss, guess, correct, correct_pred_in_batch, batch_size,
        correct_pred_in_batch / batch_size * 100))


def save_model_common(model, optimizer, epoch, train_every, test_every, all_train_losses, all_test_losses,
                      save_model_for_inference, results_path, model_name):
    if not os.path.exists(results_path):
        pathlib.Path(results_path).mkdir(parents=True)
    model_output_path = os.path.join(results_path, model_name + '.pth')
    if save_model_for_inference:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': all_train_losses,
            'all_test_losses': all_test_losses,
            'train_every': train_every,
            'test_every': test_every
        }, model_output_path)
    else:
        torch.save(model.state_dict(), model_output_path)


def save_diagram_common(all_train_losses, all_test_losses, model_name, train_every, test_every, epoch_nb, results_path):
    if not os.path.exists(results_path):
        pathlib.Path(results_path).mkdir(parents=True)
    diagram_name = model_name + '.png'
    plt.figure()
    plt.plot(list(range(train_every, epoch_nb, train_every)), all_train_losses, label='train')
    plt.plot(list(range(test_every, epoch_nb, test_every)), all_test_losses, label='test')
    plt.savefig(os.path.join(results_path, diagram_name))
    plt.legend(loc="upper right")
    plt.show()


def save_loss_common(all_train_losses, all_test_losses, model_name, results_path):
    if not os.path.exists(results_path):
        pathlib.Path(results_path).mkdir(parents=True)
    np.save(os.path.join(results_path, model_name + '_train_loss'), all_train_losses)
    np.save(os.path.join(results_path, model_name + '_test_loss'), all_test_losses)


def generate_model_name(method_name, epoch_nb, batch_size, hidden_size, learning_rate, optimizer_name):
    return '{}_ep_{}_b_{}_h_{}_lr_{}_{}'.format(method_name, epoch_nb, batch_size, hidden_size, learning_rate, optimizer_name)


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
