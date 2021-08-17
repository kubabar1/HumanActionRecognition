import time

import torch
import torch.nn as nn
import torch.optim as optim

from .model.STLSTMCell import STLSTMState
from .model.TrustGatesSTLSTMModel import TrustGatesSTLSTMModel
from .utils.STLSTMDataset import STLSTMDataset
from ...utils.training_utils import Optimizer, print_train_results, validate_model, generate_model_name, save_diagram_common, \
    save_model_common, save_loss_common


def train(classes, training_data, training_labels, validation_data, validation_labels,
          epoch_nb=2000, batch_size=128, hidden_size=128, learning_rate=0.0001,
          print_every=5, weight_decay=0.95, momentum=0.9, dropout=0.5, test_every=5,
          save_loss=True, save_diagram=True, results_path='results', optimizer_type=Optimizer.RMSPROP, save_model=True,
          save_model_for_inference=False):
    method_name = 'p_lstm_ntu'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_size = 3

    st_lstm_model = TrustGatesSTLSTMModel(input_size, hidden_size, batch_size, dropout, len(classes)).to(device)

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

    train_data_loader = STLSTMDataset(training_data, training_labels, batch_size)
    validation_data_loader = STLSTMDataset(validation_data, validation_labels, batch_size)

    for epoch in range(epoch_nb):
        data, train_y = next(iter(train_data_loader))
        tensor_train_y = torch.from_numpy(train_y).to(device)

        optimizer.zero_grad()

        tensor_train_x = torch.tensor(data, dtype=torch.float, device=device)

        joints_count = tensor_train_x.shape[2]
        spatial_dim = joints_count
        temporal_dim = data.shape[1]

        cell1_out = [[[None, None] for _ in range(spatial_dim)] for _ in range(temporal_dim)]

        losses_arr = []

        for t in range(temporal_dim):
            for j in range(spatial_dim):
                if j == 0:
                    h_spat_prev = torch.zeros(batch_size, hidden_size).to(device)
                    c_spat_prev = torch.zeros(batch_size, hidden_size).to(device)
                else:
                    h_spat_prev = cell1_out[t][j - 1][0]
                    c_spat_prev = cell1_out[t][j - 1][1]
                if t == 0:
                    h_temp_prev = torch.zeros(batch_size, hidden_size).to(device)
                    c_temp_prev = torch.zeros(batch_size, hidden_size).to(device)
                else:
                    h_temp_prev = cell1_out[t - 1][j][0]
                    c_temp_prev = cell1_out[t - 1][j][1]
                state = STLSTMState(h_temp_prev, h_spat_prev, c_temp_prev, c_spat_prev)
                input = data[:, t, j, :]
                h_next, c_next, output = st_lstm_model(torch.tensor(input, dtype=torch.float, device=device), state)

                cell1_out[t][j][0] = h_next
                cell1_out[t][j][1] = c_next

                losses_arr.append(criterion(output, tensor_train_y))

        loss = 0

        for l in losses_arr:
            loss += l

        loss /= (spatial_dim * temporal_dim)

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

                joints_count_val = tensor_val_x.shape[2]
                spatial_dim_val = joints_count_val
                temporal_dim_val = tensor_val_x.shape[1]

                cell1_out_val = [[[None, None] for _ in range(spatial_dim)] for _ in range(temporal_dim)]

                losses_arr_val = []

                for t in range(temporal_dim_val):
                    for j in range(spatial_dim_val):
                        if j == 0:
                            h_spat_prev_val = torch.zeros(batch_size, hidden_size).to(device)
                            c_spat_prev_val = torch.zeros(batch_size, hidden_size).to(device)
                        else:
                            h_spat_prev_val = cell1_out_val[t][j - 1][0]
                            c_spat_prev_val = cell1_out_val[t][j - 1][1]
                        if t == 0:
                            h_temp_prev_val = torch.zeros(batch_size, hidden_size).to(device)
                            c_temp_prev_val = torch.zeros(batch_size, hidden_size).to(device)
                        else:
                            h_temp_prev_val = cell1_out_val[t - 1][j][0]
                            c_temp_prev_val = cell1_out_val[t - 1][j][1]
                        state_val = STLSTMState(h_temp_prev_val, h_spat_prev_val, c_temp_prev_val, c_spat_prev_val)
                        input_val = data_val[:, t, j, :]
                        h_next_val, c_next_val, output_val = st_lstm_model(
                            torch.tensor(input_val, dtype=torch.float, device=device), state_val)

                        cell1_out_val[t][j][0] = h_next_val
                        cell1_out_val[t][j][1] = c_next_val

                        losses_arr_val.append(criterion(output_val, tensor_val_y))

                loss_val = 0

                for l in losses_arr_val:
                    loss_val += l

                loss_val /= (spatial_dim_val * temporal_dim_val)

                loss_val = criterion(output_val, tensor_val_y)
                test_loss_val, batch_acc_val = validate_model(tensor_val_y, output_val, classes, epoch, epoch_nb, print_every,
                                                              start_time, batch_size, loss_val)
                all_val_losses.append(test_loss_val)
                all_batch_val_accuracies.append(batch_acc_val)

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
