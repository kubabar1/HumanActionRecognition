import time

import torch
import torch.nn as nn
import torch.optim as optim

from .model.PLSTMModel import PLSTMModel
from .utils.PLSTMDataset import PLSTMDataset
from ...utils.training_utils import save_model_common, save_diagram_common, generate_model_name, print_train_results, \
    Optimizer, save_loss_common, validate_model


def train(classes, training_data, training_labels, validation_data, validation_labels,
          epoch_nb=2000, batch_size=128, hidden_size=128, learning_rate=0.00001,
          print_every=50, weight_decay=0, momentum=0.9, test_every=5, save_loss=True,
          save_diagram=True, results_path='results', optimizer_type=Optimizer.RMSPROP, save_model=True,
          save_model_for_inference=False):
    method_name = 'p_lstm_ntu'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_size = 3 * 3
    parts = 4

    p_lstm_model = PLSTMModel(input_size, hidden_size, batch_size, len(classes), parts).to(device)

    criterion = nn.NLLLoss()

    if optimizer_type == Optimizer.RMSPROP:
        optimizer = optim.RMSprop(p_lstm_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == Optimizer.SGD:
        optimizer = optim.SGD(p_lstm_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == Optimizer.ADAM:
        optimizer = optim.Adam(p_lstm_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise Exception('Unknown optimizer')

    all_train_losses = []
    all_test_losses = []

    all_batch_training_accuracies = []
    all_batch_test_accuracies = []

    start_time = time.time()
    epoch = 0

    train_data_loader = PLSTMDataset(training_data, training_labels, batch_size)
    validation_data_loader = PLSTMDataset(validation_data, validation_labels, batch_size)

    for epoch in range(epoch_nb):
        data, train_y = next(iter(train_data_loader))
        tensor_train_y = torch.from_numpy(train_y).to(device)

        optimizer.zero_grad()

        tensor_train_x = torch.tensor(data, dtype=torch.float, device=device)

        output = p_lstm_model(tensor_train_x)

        loss = criterion(output, tensor_train_y)

        loss.backward()

        optimizer.step()

        all_train_losses.append(loss.item())

        if epoch % test_every == 0 and epoch > 0:
            train_accuracy = print_train_results(classes, output, tensor_train_y, epoch, epoch_nb, start_time, loss, batch_size,
                                                 print_every)
            all_batch_training_accuracies.append(train_accuracy)
            with torch.no_grad():
                data_val, val_y = next(iter(validation_data_loader))
                tensor_test_y = torch.from_numpy(val_y).to(device)
                tensor_test_x = torch.tensor(data_val, dtype=torch.float, device=device)
                output_test = p_lstm_model(tensor_test_x)
                loss_test = criterion(output_test, tensor_test_y)
                test_loss, batch_acc = validate_model(tensor_test_y, output_test, classes, epoch, epoch_nb, print_every,
                                                      start_time, batch_size, loss_test)
                all_test_losses.append(test_loss)
                all_batch_test_accuracies.append(batch_acc)

    model_name = generate_model_name(method_name, epoch_nb, batch_size, learning_rate, optimizer_type.name, hidden_size)

    if save_diagram:
        save_diagram_common(all_train_losses, all_test_losses, model_name, test_every, epoch_nb, results_path,
                            all_batch_training_accuracies, all_batch_test_accuracies)

    if save_model:
        save_model_common(p_lstm_model, optimizer, epoch, test_every, all_train_losses, all_test_losses,
                          save_model_for_inference, results_path, model_name)

    if save_loss:
        save_loss_common(all_train_losses, all_test_losses, model_name, results_path)

    return p_lstm_model
