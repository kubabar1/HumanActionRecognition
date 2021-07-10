import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .gca_loop import gca_loop
from .utils import get_batch, classes


def train(print_every=100, plot_every=100):
    dataset_path = '../../datasets/berkeley_mhad/3d'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    iterations = 4
    batch_size = 5
    hidden_size = 128
    sequence_len = 140
    epoch_nb = 100
    analysed_kpts_left = [4, 5, 6, 11, 12, 13]
    analysed_kpts_right = [1, 2, 3, 14, 15, 16]
    all_analysed_kpts = analysed_kpts_left + analysed_kpts_right

    learning_rate = 0.0015
    decay_rate = 0.95
    momentum = 0.9
    dropout = 0.5

    criterion = nn.NLLLoss()

    optimizer = optim.SGD(lstm_rp.parameters(), lr=learning_rate, momentum=momentum)

    current_loss = 0

    all_losses = []

    start = time.time()

    for epoch in range(epoch_nb):
        data, train_y = get_data(dataset_path, batch_size, sequence_len, all_analysed_kpts)
        tensor_train_y = torch.from_numpy(train_y).to(device)

        optimizer.zero_grad()

        output = gca_loop(data, sequence_len, batch_size, hidden_size, iterations)

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

    return lstm_rp


def get_data(dataset_path, batch_size, sequence_len, analysed_kpts):
    data, labels = get_batch(dataset_path, sequence_len=sequence_len, batch_size=batch_size, training=True)
    data = data[:, :, analysed_kpts, :]
    return np.transpose(data, (1, 2, 0, 3)), labels  # [frame, joint, batch, channel]


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
