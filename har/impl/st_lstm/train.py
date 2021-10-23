import time

import torch
import torch.nn as nn
import torch.optim as optim

from .model.STLSTMModel import STLSTMModel
from .utils.STLSTMDataset import STLSTMDataset
from ...utils.dataset_util import DatasetInputType, normalise_skeleton_3d_batch
from ...utils.model_name_generator import ModelNameGenerator
from ...utils.training_utils import Optimizer, print_train_results, validate_model, save_diagram_common, \
    save_model_common, save_loss_common, get_training_batch_accuracy


def train(classes, training_data, training_labels, validation_data, validation_labels, analysed_kpts_description,
          dropout=0.5, epoch_nb=5000, batch_size=128, hidden_size=128, learning_rate=0.0001,
          weight_decay=0, momentum=0.9, val_every=5, print_every=50, lbd=0.5, steps=32, split=20, input_type=DatasetInputType.SPLIT,
          optimizer_type=Optimizer.RMSPROP, results_path='results', model_name_suffix='', save_loss=True, save_diagram=True,
          save_model=True, save_model_for_inference=False, add_random_rotation_y=False, use_two_layers=True, use_tau=False, use_bias=True,
          is_3d=True, show_diagram=True, print_results=True, use_normalization=True, add_timestamp=True):
    method_name = 'st_lstm'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_size = 3 if is_3d else 2
    joints_count = 12

    if input_type == DatasetInputType.TREE:
        joints_count = 29

    st_lstm_model = STLSTMModel(input_size, joints_count, hidden_size, len(classes), dropout, use_tau=use_tau,
                                bias=use_bias, lbd=lbd, use_two_layers=use_two_layers).to(device)

    criterion = nn.NLLLoss()

    if optimizer_type == Optimizer.RMSPROP:
        optimizer = optim.RMSprop(st_lstm_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == Optimizer.SGD:
        optimizer = optim.SGD(st_lstm_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == Optimizer.ADAM:
        optimizer = optim.Adam(st_lstm_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise Exception('Unknown optimizer')

    all_train_losses = []
    all_val_losses = []

    all_batch_training_accuracies = []
    all_batch_val_accuracies = []

    start_time = time.time()
    epoch = 0

    if use_normalization:
        training_data = normalise_skeleton_3d_batch(training_data, analysed_kpts_description['left_hip'],
                                                    analysed_kpts_description['right_hip'])
        validation_data = normalise_skeleton_3d_batch(validation_data, analysed_kpts_description['left_hip'],
                                                      analysed_kpts_description['right_hip'])

    train_data_loader = STLSTMDataset(training_data, training_labels, batch_size, analysed_kpts_description, split=split,
                                      steps=steps, input_type=input_type, add_random_rotation_y=add_random_rotation_y)
    validation_data_loader = STLSTMDataset(validation_data, validation_labels, batch_size, analysed_kpts_description,
                                           split=split, steps=steps, input_type=input_type)

    for epoch in range(epoch_nb):
        data, train_y = next(iter(train_data_loader))
        tensor_train_y = torch.from_numpy(train_y).to(device)

        optimizer.zero_grad()

        tensor_train_x = torch.tensor(data, dtype=torch.float, device=device)

        output = st_lstm_model(tensor_train_x)

        loss = criterion(output, tensor_train_y)

        loss.backward()

        optimizer.step()

        all_train_losses.append(loss.item())

        if epoch % print_every == 0 and epoch > 0 and print_results:
            print_train_results(classes, output, tensor_train_y, epoch, epoch_nb, start_time, loss, batch_size, print_every)

        if epoch % val_every == 0 and epoch > 0:
            all_batch_training_accuracies.append(get_training_batch_accuracy(classes, output, tensor_train_y, batch_size)[1])
            with torch.no_grad():
                data_val, val_y = next(iter(validation_data_loader))
                tensor_val_y = torch.from_numpy(val_y).to(device)
                tensor_val_x = torch.tensor(data_val, dtype=torch.float, device=device)
                output_val = st_lstm_model(tensor_val_x)
                loss_val = criterion(output_val, tensor_val_y)
                val_loss, batch_acc = validate_model(tensor_val_y, output_val, classes, epoch, epoch_nb, print_every,
                                                     start_time, batch_size, loss_val, print_results=print_results)
                all_val_losses.append(val_loss)
                all_batch_val_accuracies.append(batch_acc)

    model_name = ModelNameGenerator(method_name, model_name_suffix, add_timestamp) \
        .add_epoch_number(epoch_nb) \
        .add_batch_size(batch_size) \
        .add_learning_rate(learning_rate) \
        .add_optimizer_name(optimizer_type.name) \
        .add_hidden_size(hidden_size) \
        .add_input_type(input_type.name) \
        .add_dropout(dropout) \
        .add_momentum(momentum) \
        .add_weight_decay(weight_decay) \
        .add_split(split) \
        .add_steps(steps) \
        .add_lambda(lbd) \
        .add_random_rotation_y(add_random_rotation_y) \
        .add_is_bias_used(use_bias) \
        .add_is_tau(use_tau) \
        .add_is_two_layers_used(use_two_layers) \
        .add_is_3d(is_3d) \
        .add_is_normalization_used(use_normalization) \
        .generate()

    if save_model:
        save_model_common(st_lstm_model, optimizer, epoch, val_every, all_train_losses, all_val_losses,
                          save_model_for_inference, results_path, model_name)

    if save_diagram:
        save_diagram_common(all_train_losses, all_val_losses, model_name, val_every, epoch_nb, results_path,
                            all_batch_training_accuracies, all_batch_val_accuracies, show_diagram=show_diagram)

    if save_loss:
        save_loss_common(all_train_losses, all_val_losses, model_name, results_path, all_batch_training_accuracies,
                         all_batch_val_accuracies)

    return st_lstm_model
