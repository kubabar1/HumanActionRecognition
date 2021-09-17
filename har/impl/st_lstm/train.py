import time

import torch
import torch.nn as nn
import torch.optim as optim

from .model.STLSTMModel import STLSTMModel
from .utils.STLSTMDataset import STLSTMDataset
from ...utils.dataset_util import DatasetInputType
from ...utils.training_utils import Optimizer, print_train_results, validate_model, generate_model_name, save_diagram_common, \
    save_model_common, save_loss_common, get_training_batch_accuracy


def train(classes, training_data, training_labels, validation_data, validation_labels, analysed_kpts_description,
          input_size=3, dropout=0.5, epoch_nb=10000, batch_size=128, hidden_size=128, learning_rate=0.0001,
          weight_decay=0, momentum=0.9, val_every=5, lbd=0.5, input_type=DatasetInputType.SPLIT, save_loss=True,
          save_diagram=True, results_path='results', optimizer_type=Optimizer.RMSPROP, save_model=True, print_every=50,
          save_model_for_inference=False, add_random_rotation_y=False, steps=32, split=20, use_two_layers=True, use_tau=False,
          use_bias=True):
    method_name = 'st_lstm'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        if epoch % print_every == 0 and epoch > 0:
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
                                                     start_time, batch_size, loss_val)
                all_val_losses.append(val_loss)
                all_batch_val_accuracies.append(batch_acc)

    model_name = generate_model_name(method_name, epoch_nb, batch_size, learning_rate, optimizer_type.name, hidden_size,
                                     input_type.name, momentum, weight_decay, None, dropout, split, steps)

    if save_model:
        save_model_common(st_lstm_model, optimizer, epoch, val_every, all_train_losses, all_val_losses,
                          save_model_for_inference, results_path, model_name)

    if save_diagram:
        save_diagram_common(all_train_losses, all_val_losses, model_name, val_every, epoch_nb, results_path,
                            all_batch_training_accuracies, all_batch_val_accuracies)

    if save_loss:
        save_loss_common(all_train_losses, all_val_losses, model_name, results_path, all_batch_training_accuracies,
                         all_batch_val_accuracies)

    return st_lstm_model
