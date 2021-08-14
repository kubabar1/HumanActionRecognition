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


def validate_model(tensor_val_y, output_val, classes, epoch, epoch_nb, print_every, start_time, batch_size, loss_val):
    ctgs_val = [classes[int(tty)] for tty in tensor_val_y]
    gss_val = [classes[int(torch.argmax(torch.exp(o)).item())] for o in output_val]
    correct_pred_in_batch_val = len([1 for c, g in zip(ctgs_val, gss_val) if c == g])
    batch_acc = correct_pred_in_batch_val / batch_size

    if epoch % print_every == 0:
        print('VALIDATE: %d %d%% (%s) %.4f [%d/%d -> %.2f%%]' % (
            epoch, epoch / epoch_nb * 100, time_since(start_time), loss_val, correct_pred_in_batch_val, batch_size,
            batch_acc * 100))
    return loss_val.cpu().detach(), batch_acc


def print_train_results(classes, output, tensor_train_y, epoch, epoch_nb, start_time, loss, batch_size, print_every):
    ctgs = [classes[int(tty)] for tty in tensor_train_y]
    gss = [classes[int(torch.argmax(torch.exp(o)).item())] for o in output]
    correct_pred_in_batch = len([1 for c, g in zip(ctgs, gss) if c == g])
    batch_accuracy = correct_pred_in_batch / batch_size

    if epoch % print_every == 0:
        print('TRAIN: %d %d%% (%s) %.4f [%d/%d -> %.2f%%]' % (
            epoch, epoch / epoch_nb * 100, time_since(start_time), loss, correct_pred_in_batch, batch_size, batch_accuracy * 100))
    return batch_accuracy


def save_model_common(model, optimizer, epoch, train_every, validate_every, all_train_losses, all_val_losses,
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
            'all_val_losses': all_val_losses,
            'train_every': train_every,
            'validate_every': validate_every
        }, model_output_path)
    else:
        torch.save(model.state_dict(), model_output_path)


def save_diagram_common(all_train_losses, all_val_losses, model_name, validate_every, epoch_nb, results_path,
                        all_batch_training_accuracies, all_batch_val_accuracies):
    if not os.path.exists(results_path):
        pathlib.Path(results_path).mkdir(parents=True)
    diagram_name = model_name + '_loss.png'
    plt.figure()
    plt.plot(list(range(epoch_nb)), all_train_losses, label='train')
    plt.plot(list(range(validate_every, epoch_nb, validate_every)), all_val_losses, label='val')
    plt.savefig(os.path.join(results_path, diagram_name))
    plt.legend(loc="upper right")
    plt.show()

    diagram_name = model_name + '_acc.png'
    plt.figure()
    plt.plot(list(range(validate_every, epoch_nb, validate_every)), all_batch_training_accuracies, label='train')
    plt.plot(list(range(validate_every, epoch_nb, validate_every)), all_batch_val_accuracies, label='val')
    plt.savefig(os.path.join(results_path, diagram_name))
    plt.legend(loc="upper right")
    plt.show()


def save_loss_common(all_train_losses, all_val_losses, model_name, results_path, all_train_acc, all_val_acc):
    if not os.path.exists(results_path):
        pathlib.Path(results_path).mkdir(parents=True)
    np.save(os.path.join(results_path, model_name + '_train_loss'), all_train_losses)
    np.save(os.path.join(results_path, model_name + '_val_loss'), all_val_losses)
    np.save(os.path.join(results_path, model_name + '_train_acc'), all_train_acc)
    np.save(os.path.join(results_path, model_name + '_val_acc'), all_val_acc)


def generate_model_name(method_name, epoch_nb, batch_size, learning_rate, optimizer_name, hidden_size=None):
    if hidden_size:
        return '{}_ep_{}_b_{}_h_{}_lr_{}_{}'.format(method_name, epoch_nb, batch_size, hidden_size, learning_rate, optimizer_name)
    else:
        return '{}_ep_{}_b_{}_lr_{}_{}'.format(method_name, epoch_nb, batch_size, learning_rate, optimizer_name)


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
