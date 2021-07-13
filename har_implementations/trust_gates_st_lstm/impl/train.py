import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from .st_lstm.STLSTMCell import STLSTMState
from .st_lstm.TrustGatesSTLSTMModel import TrustGatesSTLSTMModel
from .utils import get_batch, classes


def train(epoch_nb=500, batch_size=5, hidden_size=128, dataset_path='../../datasets/berkeley_mhad/3d', print_every=5,
          plot_every=5, show_diagram=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_size = 3
    analysed_kpts_left = [4, 5, 6, 11, 12, 13]
    analysed_kpts_right = [1, 2, 3, 14, 15, 16]
    all_analysed_kpts = analysed_kpts_left + analysed_kpts_right

    learning_rate = 0.002
    momentum = 0.9
    weight_decay = 0.95
    dropout = 0.5

    st_lstm_model = TrustGatesSTLSTMModel(input_size, hidden_size, batch_size, dropout, len(classes)).to(device)

    criterion = nn.NLLLoss()

    # optimizer = optim.RMSprop(st_lstm_model.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = optim.SGD(st_lstm_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # optimizer = optim.Adam(st_lstm_model.parameters(), lr=learning_rate)

    current_loss = 0

    all_losses = []

    start = time.time()

    for epoch in range(epoch_nb):
        data, train_y = get_data(dataset_path, batch_size, all_analysed_kpts)
        tensor_train_y = torch.from_numpy(np.array(train_y)).to(device)

        optimizer.zero_grad()

        joints_count = data.shape[1]
        spatial_dim = joints_count
        temporal_dim = data.shape[0]

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
                input = data[t][j]
                h_next, c_next, output = st_lstm_model(torch.tensor(input, dtype=torch.float, device=device), state)

                cell1_out[t][j][0] = h_next
                cell1_out[t][j][1] = c_next

                losses_arr.append(criterion(output, tensor_train_y))

        # data = np.transpose(data, (2, 0, 1, 3))
        # data = data.reshape((batch_size, 20, -1))
        # output = st_lstm_model(torch.tensor(data, dtype=torch.float, device=device), device)

        # print(output)
        # print(tensor_train_y)
        # print(tensor_train_y.shape)

        loss = 0

        for l in losses_arr:
            loss += l

        loss /= (spatial_dim * temporal_dim)

        loss.backward()

        optimizer.step()

        current_loss += loss.item()

        if epoch % print_every == 0 and epoch > 0:
            category = classes[int(tensor_train_y[0])]

            guess = classes[int(torch.argmax(torch.exp(output)[0]).item())]

            correct = '✓' if guess == category else '✗ (%s)' % category

            print(' %d %d%% (%s) %.4f  / %s %s' % (epoch, epoch / epoch_nb * 100, time_since(start), loss, guess, correct))

        # Add current loss avg to list of losses
        if epoch % plot_every == 0 and epoch > 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    if show_diagram:
        plt.figure()
        plt.plot(list(range(plot_every, epoch_nb, plot_every)), all_losses)
        plt.show()

    return st_lstm_model


def get_data(dataset_path, batch_size, analysed_kpts):
    data, labels = get_batch(dataset_path, batch_size=batch_size, training=True)
    data = data[:, :, analysed_kpts, :]
    return np.transpose(data, (1, 2, 0, 3)), labels  # [frame, joint, batch, channels]


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
