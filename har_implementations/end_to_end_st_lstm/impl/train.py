import math
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

from .model.MainLSTMNNModel import MainLSTMNNModel
from .utils import get_batch, get_batch2, classes


def train(epoch_nb=2000, sequence_len=100, batch_size=64, hidden_size=128, dataset_path='../../datasets/berkeley_mhad/3d',
          print_every=25, plot_every=25, show_diagram=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_size = 3 * 12
    analysed_kpts_left = [4, 5, 6, 11, 12, 13]
    analysed_kpts_right = [1, 2, 3, 14, 15, 16]
    # all_analysed_kpts = analysed_kpts_left + analysed_kpts_right
    all_analysed_kpts = [16, 15, 14, 13, 12, 11, 3, 2, 1, 6, 5, 4]

    learning_rate = 0.0001  # 0.001
    weight_decay = 0.95
    momentum = 0.9
    dropout = 0.5

    # spatial_model = SpatialAttentionModel(input_size, hidden_size, batch_size, len(classes)).to(device)
    main_model = MainLSTMNNModel(input_size, hidden_size, batch_size, len(classes)).to(device)

    criterion = nn.NLLLoss()

    # optimizer = optim.RMSprop(main_model.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = optim.SGD(main_model.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = optim.Adam(main_model.parameters(), lr=learning_rate)

    current_loss = 0

    all_losses = []

    start = time.time()

    np.set_printoptions(suppress=True)
    torch.set_printoptions(sci_mode=False)

    lambda_1 = 0.05  # 0.001
    lambda_2 = 0.001  # 0.0001
    lambda_3 = 0.0000005  # 0.0005

    # data, train_y = get_data(dataset_path, batch_size, all_analysed_kpts, sequence_len)
    # data = torch.tensor(data, dtype=torch.float, device=device)
    # alpha_arr = spatial_model(data)

    for epoch in range(epoch_nb):
        data, train_y = get_data(dataset_path, batch_size, all_analysed_kpts, sequence_len)
        tensor_train_y = torch.from_numpy(np.array(train_y)).to(device)

        optimizer.zero_grad()

        data = data.reshape((batch_size, 20, -1))
        data = torch.tensor(data, dtype=torch.float, device=device)

        output, alpha_arr, beta_arr = main_model(data, epoch)
        loss = criterion(output, tensor_train_y)

        _, T, K = alpha_arr.shape

        l1 = 0
        for k in range(K):
            tmp = 0
            for t in range(T):
                tmp += alpha_arr[:, t, k]
            tmp /= T
            l1 += (1 - tmp) ** 2

        l2 = 0
        for t in range(T):
            l2 += torch.norm(beta_arr[:, t, :], 2, -1)

        l3 = 0
        for param in main_model.parameters():
            l3 += torch.norm(param, 1)

        # print(loss)

        # print(beta_arr.shape)

        # print('##############################################################')
        # print(lambda_1 * torch.mean(l1))
        # print((lambda_2 / T) * torch.mean(l2))
        # print(lambda_3 * l3)
        # print('##############################################################')

        loss += lambda_1 * torch.mean(l1) + (lambda_2 / T) * torch.mean(l2) + lambda_3 * l3

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
            # tmp = np.array([t[0, :].detach().cpu().numpy() for t in alpha_arr], dtype=float)
            # print(np.sum(tmp, axis=0))
            print(torch.sum(alpha_arr[0], dim=0))
            print(torch.mean(beta_arr[0]))

            category = classes[int(tensor_train_y[1])]

            guess = classes[int(torch.argmax(torch.exp(output)[1]).item())]

            correct = '✓' if guess == category else '✗ (%s)' % category

            # if not epoch % 100:
            #     window_name = 'Image'
            #     image = cv2.imread('/home/kuba/Downloads/empty.png')
            #     radius = 10
            #     center_coordinates = (120, 50)
            #     color = (255, 0, 0)
            #     thickness = 2
            #     for i in range(12):
            #         image = cv2.circle(image, (int((data[0][0][i * 3] + 1)/2 * 255),
            #                                    int((data[0][0][i * 3 + 1] + 1)/2 * 255)), radius, color, thickness)
            #
            #     cv2.imshow(window_name, image)
            #     k = cv2.waitKey(25) & 0xFF
            #     if k == ord('q'):
            #         cv2.destroyAllWindows()

            print(' %d %d%% (%s) %.4f  / %s %s' % (epoch, epoch / epoch_nb * 100, time_since(start), loss, guess, correct))
            # tmp = np.array([t[0, :].detach().cpu().numpy() for t in alpha_arr], dtype=float)
            # print(np.sum(tmp, axis=0))
            print(torch.sum(alpha_arr[1], dim=0))
            print(torch.mean(beta_arr[1]))

            print(all_analysed_kpts)

        # Add current loss avg to list of losses
        if epoch % plot_every == 0 and epoch > 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    if show_diagram:
        plt.figure()
        plt.plot(list(range(plot_every, epoch_nb, plot_every)), all_losses)
        plt.show()

    return main_model


def get_data(dataset_path, batch_size, analysed_kpts, sequence_len):
    data, labels = get_batch2(dataset_path, batch_size=batch_size, training=True)  # , sequence_len=sequence_len
    return data[:, :, analysed_kpts, :], labels


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
