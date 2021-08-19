import time

import torch
import torch.nn as nn
import torch.optim as optim

from .model.TrustGatesSTLSTMModel import TrustGatesSTLSTMModel
from .utils.STLSTMDataset import STLSTMDataset
from ...utils.training_utils import Optimizer, print_train_results, validate_model, generate_model_name, save_diagram_common, \
    save_model_common, save_loss_common


def train(classes, training_data, training_labels, validation_data, validation_labels,
          epoch_nb=2000, batch_size=128, hidden_size=256, learning_rate=0.00001,
          print_every=10, weight_decay=0, momentum=0.9, dropout=0.5, test_every=5, split_data=20,
          save_loss=True, save_diagram=True, results_path='results', optimizer_type=Optimizer.RMSPROP, save_model=True,
          save_model_for_inference=False):
    method_name = 'st_lstm'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # weight_decay = 0.95
    input_size = 3
    joints_count = 12
    spatial_dim = joints_count
    temporal_dim = split_data

    criterion = nn.NLLLoss()

    st_lstm_model = TrustGatesSTLSTMModel(input_size, hidden_size, batch_size, len(classes), spatial_dim, temporal_dim, criterion,
                                          dropout).to(device)

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

    train_data_loader = STLSTMDataset(training_data, training_labels, batch_size, split=split_data)
    validation_data_loader = STLSTMDataset(validation_data, validation_labels, batch_size, split=split_data)

    for epoch in range(epoch_nb):
        data, train_y = next(iter(train_data_loader))
        tensor_train_y = torch.from_numpy(train_y).to(device)

        optimizer.zero_grad()

        tensor_train_x = torch.tensor(data, dtype=torch.float, device=device)

        output, loss = st_lstm_model(tensor_train_x, tensor_train_y)

        loss.backward()

        optimizer.step()

        all_train_losses.append(loss.item())

        if epoch % test_every == 0 and epoch > 0:
            train_accuracy = print_train_results(classes, output, tensor_train_y, epoch, epoch_nb, start_time, loss, batch_size,
                                                 print_every)
            all_batch_training_accuracies.append(train_accuracy)
            with torch.no_grad():
                data_val, val_y = next(iter(validation_data_loader))
                tensor_val_y = torch.from_numpy(val_y).to(device)
                tensor_val_x = torch.tensor(data_val, dtype=torch.float, device=device)
                output_val, loss_val = st_lstm_model(tensor_val_x, tensor_val_y)
                val_loss, batch_acc = validate_model(tensor_val_y, output_val, classes, epoch, epoch_nb, print_every,
                                                     start_time, batch_size, loss_val)
                all_val_losses.append(val_loss)
                all_batch_val_accuracies.append(batch_acc)

    model_name = generate_model_name(method_name, epoch_nb, batch_size, learning_rate, optimizer_type.name, hidden_size)

    if save_diagram:
        save_diagram_common(all_train_losses, all_val_losses, model_name, test_every, epoch_nb, results_path,
                            all_batch_training_accuracies, all_batch_val_accuracies)

    if save_model:
        save_model_common(st_lstm_model, optimizer, epoch, test_every, all_train_losses, all_val_losses,
                          save_model_for_inference, results_path, model_name)

    if save_loss:
        save_loss_common(all_train_losses, all_val_losses, model_name, results_path, all_batch_training_accuracies,
                         all_batch_val_accuracies)

    return st_lstm_model
