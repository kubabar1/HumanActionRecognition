import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from .model.HierarchicalRNNModel import HierarchicalRNNModel
from .utils import classes, video_pose_3d_kpts, get_batch2


def train(epoch_nb=1000, batch_size=32, hidden_size=64, dataset_path='../../datasets/berkeley_mhad/3d', print_every=25,
          plot_every=25, show_diagram=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    learning_rate = 0.0001
    weight_decay = 0.95
    momentum = 0.9
    dropout = 0.5

    st_lstm_model = HierarchicalRNNModel(hidden_size, batch_size, len(classes)).to(device)

    criterion = nn.NLLLoss()

    optimizer = optim.RMSprop(st_lstm_model.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = optim.SGD(st_lstm_model.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = optim.Adam(st_lstm_model.parameters(), lr=learning_rate)

    current_loss = 0

    all_losses = []

    start = time.time()

    for epoch in range(epoch_nb):
        data, train_y = get_data(dataset_path, batch_size)
        tensor_train_y = torch.from_numpy(np.array(train_y)).to(device)

        optimizer.zero_grad()

        output = st_lstm_model(torch.tensor(data, dtype=torch.float, device=device))

        loss = criterion(output, tensor_train_y)

        loss.backward()

        optimizer.step()

        current_loss += loss.item()

        if epoch % print_every == 0 and epoch > 0:
            ctgs = [classes[int(tty)] for tty in tensor_train_y]
            gss = [classes[int(torch.argmax(torch.exp(o)).item())] for o in output]
            correct_pred_in_batch = len([1 for c, g in zip(ctgs, gss) if c == g])

            category = classes[int(tensor_train_y[0])]

            guess = classes[int(torch.argmax(torch.exp(output)[0]).item())]

            correct = '✓' if guess == category else '✗ (%s)' % category

            print(' %d %d%% (%s) %.4f  / %s %s [%d/%d -> %.2f%%]' % (
                epoch, epoch / epoch_nb * 100, time_since(start), loss, guess, correct, correct_pred_in_batch, batch_size,
                correct_pred_in_batch / batch_size * 100))

        # Add current loss avg to list of losses
        if epoch % plot_every == 0 and epoch > 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    if show_diagram:
        plt.figure()
        plt.plot(list(range(plot_every, epoch_nb, plot_every)), all_losses)
        plt.show()

    return st_lstm_model


def get_data(dataset_path, batch_size):
    data, labels = get_batch(dataset_path, batch_size=batch_size, training=True)
    right_wrists = data[:, :, video_pose_3d_kpts['right_wrist'], :]
    left_wrists = data[:, :, video_pose_3d_kpts['left_wrist'], :]
    right_elbows = data[:, :, video_pose_3d_kpts['right_elbow'], :]
    left_elbows = data[:, :, video_pose_3d_kpts['left_elbow'], :]
    right_shoulders = data[:, :, video_pose_3d_kpts['right_shoulder'], :]
    left_shoulders = data[:, :, video_pose_3d_kpts['left_shoulder'], :]
    right_hips = data[:, :, video_pose_3d_kpts['right_hip'], :]
    left_hips = data[:, :, video_pose_3d_kpts['left_hip'], :]
    right_knees = data[:, :, video_pose_3d_kpts['right_knee'], :]
    left_knees = data[:, :, video_pose_3d_kpts['left_knee'], :]
    right_ankles = data[:, :, video_pose_3d_kpts['right_ankle'], :]
    left_ankles = data[:, :, video_pose_3d_kpts['left_ankle'], :]

    left_arms = np.concatenate((left_wrists, left_elbows, left_shoulders), axis=2)
    right_arms = np.concatenate((right_wrists, right_elbows, right_shoulders), axis=2)
    left_legs = np.concatenate((left_hips, left_knees, left_ankles), axis=2)
    right_legs = np.concatenate((right_hips, right_knees, right_ankles), axis=2)

    return [left_arms, right_arms, left_legs, right_legs], labels  # [batch, time_sequence, joint_channels]


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
