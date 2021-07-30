import time

import torch
import torch.nn as nn
import torch.optim as optim

from .model.LSTMSimpleModel import LSTMSimpleModel
from ...utils.training_utils import save_model_common, save_diagram_common, generate_model_name, print_train_results, \
    Optimizer, save_loss_common, test_model


def train(classes, get_batch, dataset_path, epoch_nb=5000, batch_size=128, hidden_size=128, learning_rate=0.00001,
          print_train_every=50, print_test_every=50, weight_decay=0, momentum=0.9, train_every=10, test_every=5, save_loss=True,
          save_diagram=True, results_path='results', optimizer_type=Optimizer.RMSPROP, save_model=True,
          save_model_for_inference=False):
    method_name = 'lstm_simple'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_size = 3 * 12

    st_lstm_model = LSTMSimpleModel(input_size, hidden_size, batch_size, len(classes)).to(device)

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
    all_test_losses = []

    start_time = time.time()
    current_train_loss = 0
    epoch = 0

    for epoch in range(epoch_nb):
        data, train_y = get_batch(dataset_path, batch_size)
        tensor_train_y = torch.from_numpy(train_y).to(device)

        optimizer.zero_grad()

        tensor_train_x = torch.tensor(data.reshape((data.shape[0], data.shape[1], -1)), dtype=torch.float, device=device)

        output = st_lstm_model(tensor_train_x)

        loss = criterion(output, tensor_train_y)

        loss.backward()

        optimizer.step()

        current_train_loss += loss.item()

        if epoch % print_train_every == 0 and epoch > 0:
            print_train_results(classes, output, tensor_train_y, epoch, epoch_nb, start_time, loss, batch_size)

        if epoch % test_every == 0 and epoch > 0:
            test_model(st_lstm_model, criterion, classes, get_batch, dataset_path, batch_size, device, epoch, epoch_nb,
                       print_test_every, all_test_losses, start_time)

        if epoch % train_every == 0 and epoch > 0:
            all_train_losses.append(current_train_loss / train_every)
            current_train_loss = 0

    model_name = generate_model_name(method_name, epoch_nb, batch_size, hidden_size, learning_rate, optimizer_type.name)

    if save_diagram:
        save_diagram_common(all_train_losses, all_test_losses, model_name, train_every, test_every, epoch_nb, results_path)

    if save_model:
        save_model_common(st_lstm_model, optimizer, epoch, train_every, test_every, all_train_losses, all_test_losses,
                          save_model_for_inference, results_path, model_name)

    if save_loss:
        save_loss_common(all_train_losses, all_test_losses, model_name, results_path)

    return st_lstm_model
