import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .st_lstm.LSTMSimpleModel import LSTMSimpleModel
from .utils import get_batch, classes


def train(epoch_nb=10000, batch_size=5, hidden_size=128, dataset_path='../../datasets/berkeley_mhad/3d', print_every=100,
          plot_every=100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_size = 3 * 12
    analysed_kpts_left = [4, 5, 6, 11, 12, 13]
    analysed_kpts_right = [1, 2, 3, 14, 15, 16]
    all_analysed_kpts = analysed_kpts_left + analysed_kpts_right

    learning_rate = 0.0015
    weight_decay = 0.95
    momentum = 0.9
    dropout = 0.5

    st_lstm_model = LSTMSimpleModel(input_size, hidden_size, batch_size, len(classes)).to(device)

    criterion = nn.NLLLoss()

    optimizer = optim.RMSprop(st_lstm_model.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = optim.SGD(st_lstm_model.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = optim.Adam(st_lstm_model.parameters(), lr=learning_rate)

    current_loss = 0

    all_losses = []

    start = time.time()

    for epoch in range(epoch_nb):
        data, train_y = get_data(dataset_path, batch_size, all_analysed_kpts)
        tensor_train_y = torch.from_numpy(np.array(train_y)).to(device)

        optimizer.zero_grad()

        data = np.transpose(data, (2, 0, 1, 3))
        data = data.reshape((batch_size, 20, -1))
        output = st_lstm_model(torch.tensor(data, dtype=torch.float, device=device))

        loss = criterion(output, tensor_train_y)

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
