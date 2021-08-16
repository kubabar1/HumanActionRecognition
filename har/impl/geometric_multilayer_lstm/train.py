import time

import torch
import torch.nn as nn
import torch.optim as optim

from .model.MultilayerLSTM import MultilayerLSTM
from .utils.MultilayerLSTMDataset import MultilayerLSTMDataset
from ...utils.dataset_util import SetType
from ...utils.training_utils import save_model_common, save_diagram_common, generate_model_name, print_train_results, \
    Optimizer, save_loss_common, validate_model


def train(classes, training_data, training_labels, validation_data, validation_labels,
          epoch_nb=5000, batch_size=128, hidden_size=128, learning_rate=0.000005,
          print_every=25, weight_decay=0, momentum=0.9, val_every=5, save_loss=True,
          save_diagram=True, results_path='results', optimizer_type=Optimizer.RMSPROP, save_model=True,
          save_model_for_inference=False, use_cache=True):
    method_name = 'geometric_multilayer_lstm'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lstm_model_rp = MultilayerLSTM(396, hidden_size, len(classes)).to(device)
    lstm_model_jjd = MultilayerLSTM(132, hidden_size, len(classes)).to(device)
    lstm_model_jld = MultilayerLSTM(180, hidden_size, len(classes)).to(device)

    criterion_rp = nn.NLLLoss()
    criterion_jjd = nn.NLLLoss()
    criterion_jld = nn.NLLLoss()

    if optimizer_type == Optimizer.RMSPROP:
        optimizer_rp = optim.RMSprop(lstm_model_rp.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        optimizer_jjd = optim.RMSprop(lstm_model_jjd.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        optimizer_jld = optim.RMSprop(lstm_model_jld.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == Optimizer.SGD:
        optimizer_rp = optim.SGD(lstm_model_rp.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        optimizer_jjd = optim.SGD(lstm_model_jjd.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        optimizer_jld = optim.SGD(lstm_model_jld.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == Optimizer.ADAM:
        optimizer_rp = optim.Adam(lstm_model_rp.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer_jjd = optim.Adam(lstm_model_jjd.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer_jld = optim.Adam(lstm_model_jld.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise Exception('Unknown optimizer')

    all_train_losses_rp = []
    all_train_losses_jjd = []
    all_train_losses_jld = []

    all_val_losses_rp = []
    all_val_losses_jjd = []
    all_val_losses_jld = []

    all_batch_training_accuracies_rp = []
    all_batch_training_accuracies_jjd = []
    all_batch_training_accuracies_jld = []

    all_batch_val_accuracies_rp = []
    all_batch_val_accuracies_jjd = []
    all_batch_val_accuracies_jld = []

    start_time = time.time()
    epoch = 0

    train_data_loader = MultilayerLSTMDataset(training_data, training_labels, batch_size, SetType.TRAINING, use_cache=use_cache)
    validation_data_loader = MultilayerLSTMDataset(validation_data, validation_labels, batch_size, SetType.VALIDATION,
                                                   use_cache=use_cache)

    for epoch in range(epoch_nb):
        data, train_y = next(iter(train_data_loader))
        tensor_train_y = torch.from_numpy(train_y).to(device)

        tensor_train_x_rp = torch.tensor(data[0], dtype=torch.float, device=device)
        tensor_train_x_jjd = torch.tensor(data[1], dtype=torch.float, device=device)
        tensor_train_x_jld = torch.tensor(data[2], dtype=torch.float, device=device)

        optimizer_rp.zero_grad()
        optimizer_jjd.zero_grad()
        optimizer_jld.zero_grad()

        output_rp = lstm_model_rp(tensor_train_x_rp)
        output_jjd = lstm_model_jjd(tensor_train_x_jjd)
        output_jld = lstm_model_jld(tensor_train_x_jld)

        loss_rp = criterion_rp(output_rp, tensor_train_y)
        loss_jjd = criterion_jjd(output_jjd, tensor_train_y)
        loss_jld = criterion_jld(output_jld, tensor_train_y)

        loss_rp.backward()
        loss_jjd.backward()
        loss_jld.backward()

        optimizer_rp.step()
        optimizer_jjd.step()
        optimizer_jld.step()

        all_train_losses_rp.append(loss_rp.item())
        all_train_losses_jjd.append(loss_jjd.item())
        all_train_losses_jld.append(loss_jld.item())

        if epoch % val_every == 0 and epoch > 0:
            train_accuracy_rp = print_train_results(classes, output_rp, tensor_train_y, epoch, epoch_nb, start_time, loss_rp,
                                                    batch_size, print_every, 'RP')
            train_accuracy_jjd = print_train_results(classes, output_jjd, tensor_train_y, epoch, epoch_nb, start_time, loss_jjd,
                                                     batch_size, print_every, 'JJD')
            train_accuracy_jld = print_train_results(classes, output_jld, tensor_train_y, epoch, epoch_nb, start_time, loss_jld,
                                                     batch_size, print_every, 'JLD')

            all_batch_training_accuracies_rp.append(train_accuracy_rp)
            all_batch_training_accuracies_jjd.append(train_accuracy_jjd)
            all_batch_training_accuracies_jld.append(train_accuracy_jld)
            with torch.no_grad():
                data_val, val_y = next(iter(validation_data_loader))
                tensor_val_y = torch.from_numpy(val_y).to(device)

                data_val_rp = data_val[0]
                data_val_jjd = data_val[1]
                data_val_jld = data_val[2]

                tensor_val_x_rp = torch.tensor(data_val_rp, dtype=torch.float, device=device)
                tensor_val_x_jjd = torch.tensor(data_val_jjd, dtype=torch.float, device=device)
                tensor_val_x_jld = torch.tensor(data_val_jld, dtype=torch.float, device=device)

                output_val_rp = lstm_model_rp(tensor_val_x_rp)
                output_val_jjd = lstm_model_jjd(tensor_val_x_jjd)
                output_val_jld = lstm_model_jld(tensor_val_x_jld)

                loss_val_rp = criterion_rp(output_val_rp, tensor_val_y)
                loss_val_jjd = criterion_jjd(output_val_jjd, tensor_val_y)
                loss_val_jld = criterion_jld(output_val_jld, tensor_val_y)

                val_loss_rp, batch_acc_rp = validate_model(tensor_val_y, output_val_rp, classes, epoch, epoch_nb, print_every,
                                                           start_time, batch_size, loss_val_rp, 'RP')
                val_loss_jjd, batch_acc_jjd = validate_model(tensor_val_y, output_val_jjd, classes, epoch, epoch_nb, print_every,
                                                             start_time, batch_size, loss_val_jjd, 'JJD')
                val_loss_jld, batch_acc_jld = validate_model(tensor_val_y, output_val_jld, classes, epoch, epoch_nb, print_every,
                                                             start_time, batch_size, loss_val_jld, 'JLD')

                all_val_losses_rp.append(val_loss_rp)
                all_val_losses_jjd.append(val_loss_jjd)
                all_val_losses_jld.append(val_loss_jld)

                all_batch_val_accuracies_rp.append(batch_acc_rp)
                all_batch_val_accuracies_jjd.append(batch_acc_jjd)
                all_batch_val_accuracies_jld.append(batch_acc_jld)

    model_name = generate_model_name(method_name, epoch_nb, batch_size, learning_rate, optimizer_type.name, hidden_size)

    model_name_rp = model_name + '_rp'
    model_name_jjd = model_name + '_jjd'
    model_name_jld = model_name + '_jld'

    if save_diagram:
        save_diagram_common(all_train_losses_rp, all_val_losses_rp, model_name_rp, val_every, epoch_nb, results_path,
                            all_batch_training_accuracies_rp, all_batch_val_accuracies_rp)
        save_diagram_common(all_train_losses_jjd, all_val_losses_jjd, model_name_jjd, val_every, epoch_nb, results_path,
                            all_batch_training_accuracies_jjd, all_batch_val_accuracies_jjd)
        save_diagram_common(all_train_losses_jld, all_val_losses_jld, model_name_jld, val_every, epoch_nb, results_path,
                            all_batch_training_accuracies_jld, all_batch_val_accuracies_jld)

    if save_model:
        save_model_common(lstm_model_rp, optimizer_rp, epoch, val_every, all_train_losses_rp, all_val_losses_rp,
                          save_model_for_inference, results_path, model_name_rp)
        save_model_common(lstm_model_jjd, optimizer_jjd, epoch, val_every, all_train_losses_jjd, all_val_losses_jjd,
                          save_model_for_inference, results_path, model_name_jjd)
        save_model_common(lstm_model_jld, optimizer_jld, epoch, val_every, all_train_losses_jld, all_val_losses_jld,
                          save_model_for_inference, results_path, model_name_jld)

    if save_loss:
        save_loss_common(all_train_losses_rp, all_val_losses_rp, model_name_rp, results_path, all_batch_training_accuracies_rp,
                         all_batch_val_accuracies_rp)
        save_loss_common(all_train_losses_jjd, all_val_losses_jjd, model_name_jjd, results_path,
                         all_batch_training_accuracies_jjd, all_batch_val_accuracies_jjd)
        save_loss_common(all_train_losses_jld, all_val_losses_jld, model_name_jld, results_path,
                         all_batch_training_accuracies_jld, all_batch_val_accuracies_jld)

    return lstm_model_rp, lstm_model_jjd, lstm_model_jld
