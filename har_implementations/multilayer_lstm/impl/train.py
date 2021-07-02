import math
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from .LSTM import LSTM
from .utils import prepare_data_berkeley_mhad, get_batch, classes


def train(dataset_path, recreate_batch_dataset=False, batch_cache_path='batch_cache', n_iters=10000, print_every=100,
          plot_every=100, batch_size=128, time_stamp=20):
    n_layer = 3
    n_categories = len(classes)
    learning_rate = 0.0005
    momentum = 0.9
    hidden_dim = batch_size

    analysed_kpts = [16, 15, 14, 11, 12, 13, 1, 2, 3, 4, 5, 6]
    analysed_lines_1 = [[3, 2], [2, 1], [16, 15], [15, 14], [12, 11], [13, 12], [5, 4], [6, 5]]
    analysed_lines_2 = [[3, 1], [16, 14], [13, 11], [6, 4]]
    analysed_lines_3 = [[3, 16], [6, 16], [3, 6], [16, 6], [13, 6], [16, 13]]
    analysed_lines = analysed_lines_1 + analysed_lines_2 + analysed_lines_3

    if not os.path.exists(batch_cache_path) or recreate_batch_dataset:
        prepare_data_berkeley_mhad(dataset_path, classes, analysed_lines, analysed_kpts)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_rp, train_jjd, train_jld, _ = get_batch(batch_size=1, batch_cache_path=batch_cache_path)
    input_size_rp = train_rp.shape[2]
    input_size_jjd = train_jjd.shape[2]
    input_size_jld = train_jld.shape[2]

    lstm_rp = LSTM(input_size_rp, hidden_dim, n_categories, n_layer, time_stamp)
    lstm_jjd = LSTM(input_size_jjd, hidden_dim, n_categories, n_layer, time_stamp)
    lstm_jld = LSTM(input_size_jld, hidden_dim, n_categories, n_layer, time_stamp)

    lstm_rp.to(device)
    lstm_jjd.to(device)
    lstm_jld.to(device)

    criterion_rp = nn.NLLLoss()
    criterion_jjd = nn.NLLLoss()
    criterion_jld = nn.NLLLoss()

    optimizer_rp = optim.RMSprop(lstm_rp.parameters(), lr=learning_rate, momentum=momentum)
    optimizer_jjd = optim.RMSprop(lstm_jjd.parameters(), lr=learning_rate, momentum=momentum)
    optimizer_jld = optim.RMSprop(lstm_jld.parameters(), lr=learning_rate, momentum=momentum)

    current_loss_rp = 0
    current_loss_jjd = 0
    current_loss_jld = 0

    all_losses_rp = []
    all_losses_jjd = []
    all_losses_jld = []

    start = time.time()

    for epoch in range(n_iters):
        train_rp_x, train_jjd_x, train_jld_x, train_y = get_batch(batch_size=batch_size, batch_cache_path=batch_cache_path)
        tensor_train_x_rp = torch.from_numpy(train_rp_x).float()
        tensor_train_x_jjd = torch.from_numpy(train_jjd_x).float()
        tensor_train_x_jld = torch.from_numpy(train_jld_x).float()
        tensor_train_y = torch.from_numpy(train_y)

        tensor_train_x_rp = tensor_train_x_rp.to(device)
        tensor_train_x_jjd = tensor_train_x_jjd.to(device)
        tensor_train_x_jld = tensor_train_x_jld.to(device)
        tensor_train_y = tensor_train_y.to(device)

        optimizer_rp.zero_grad()
        optimizer_jjd.zero_grad()
        optimizer_jld.zero_grad()

        output_rp = lstm_rp(tensor_train_x_rp)
        output_jjd = lstm_jjd(tensor_train_x_jjd)
        output_jld = lstm_jld(tensor_train_x_jld)

        loss_rp = criterion_rp(output_rp, tensor_train_y)
        loss_jjd = criterion_jjd(output_jjd, tensor_train_y)
        loss_jld = criterion_jld(output_jld, tensor_train_y)

        loss_rp.backward()
        loss_jjd.backward()
        loss_jld.backward()

        optimizer_rp.step()
        optimizer_jjd.step()
        optimizer_jld.step()

        # scheduler.step()

        current_loss_rp += loss_rp.item()
        current_loss_jjd += loss_jjd.item()
        current_loss_jld += loss_jld.item()

        # Print iter number, loss, name and guess
        if epoch % print_every == 0 and epoch > 0:
            category = classes[int(tensor_train_y[0])]

            guess_rp = classes[int(torch.argmax(torch.exp(output_rp)[0]).item())]
            guess_jjd = classes[int(torch.argmax(torch.exp(output_jjd)[0]).item())]
            guess_jld = classes[int(torch.argmax(torch.exp(output_jld)[0]).item())]

            correct_rp = '✓' if guess_rp == category else '✗ (%s)' % category
            correct_jjd = '✓' if guess_jjd == category else '✗ (%s)' % category
            correct_jld = '✓' if guess_jld == category else '✗ (%s)' % category

            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('RP: %d %d%% (%s) %.4f  / %s %s' % (
                epoch, epoch / n_iters * 100, time_since(start), loss_rp, guess_rp, correct_rp))
            print('JJD: %d %d%% (%s) %.4f  / %s %s' % (
                epoch, epoch / n_iters * 100, time_since(start), loss_jjd, guess_jjd, correct_jjd))
            print('JLD: %d %d%% (%s) %.4f  / %s %s' % (
                epoch, epoch / n_iters * 100, time_since(start), loss_jld, guess_jld, correct_jld))
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        # Add current loss avg to list of losses
        if epoch % plot_every == 0 and epoch > 0:
            all_losses_rp.append(current_loss_rp / plot_every)
            all_losses_jjd.append(current_loss_jjd / plot_every)
            all_losses_jld.append(current_loss_jld / plot_every)
            current_loss_rp = 0
            current_loss_jjd = 0
            current_loss_jld = 0

    return lstm_rp, lstm_jjd, lstm_jld


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
