from configparser import ConfigParser
import os
import numpy as np
import torch
import random
import torch.optim as optim
import torch.nn as nn
import time
import math

from LSTM import LSTM


def main():
    # Output classes to learn how to classify
    LABELS = [
        "JUMPING_IN_PLACE",
        "JUMPING_JACKS",
        "BENDING_HANDS_UP_ALL_THE_WAY_DOWN",
        "PUNCHING_BOXING",
        "WAVING_TWO_HANDS",
        "WAVING_ONE_HAND_RIGHT",
        "CLAPPING_HANDS",
        "THROWING_A_BALL",
        "SIT_DOWN_THEN_STAND_UP",
        "SIT_DOWN",
        "STAND_UP",
        # "T_POSE"
    ]

    coordinates_path = './data/coordinates'

    n_steps = 32  # 32 timesteps per series
    n_joints = 34
    n_categories = len(LABELS)

    x_train, y_train, x_test, y_test = prepare_data(coordinates_path, n_steps, n_joints)

    tensor_x_test = torch.from_numpy(x_test)
    print('test_data_size:', tensor_x_test.size())
    tensor_y_test = torch.from_numpy(y_test)
    print('test_label_size:', tensor_y_test.size())
    n_data_size_test = tensor_x_test.size()[0]
    print('n_data_size_test:', n_data_size_test)

    tensor_x_train = torch.from_numpy(x_train)
    print('train_data_size:', tensor_x_train.size())
    tensor_y_train = torch.from_numpy(y_train)
    print('train_label_size:', tensor_y_train.size())
    n_data_size_train = tensor_x_train.size()[0]
    print('n_data_size_train:', n_data_size_train)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    n_hidden = 128
    n_layer = 3
    rnn = LSTM(n_joints, n_hidden, n_categories, n_layer, n_steps)
    rnn.to(device)

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.0005
    optimizer = optim.SGD(rnn.parameters(), lr=learning_rate, momentum=0.9)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

    n_iters = 100000
    # n_iters = 60000
    print_every = 1000
    plot_every = 1000
    batch_size = 128

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    start = time.time()

    for iter in range(1, n_iters + 1):
        category_tensor, input_sequence = random_training_example_batch(tensor_x_train, tensor_y_train,
                                                                        n_data_size_train, tensor_x_test, tensor_y_test,
                                                                        n_data_size_test, batch_size, 'train')
        input_sequence = input_sequence.to(device)
        category_tensor = category_tensor.to(device)
        category_tensor = torch.squeeze(category_tensor)

        optimizer.zero_grad()

        output = rnn(input_sequence)
        loss = criterion(output, category_tensor)
        loss.backward()
        optimizer.step()
        # scheduler.step()

        current_loss += loss.item()

        category = LABELS[int(category_tensor[0])]

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = category_from_output(output, LABELS)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f  / %s %s' % (iter, iter / n_iters * 100, time_since(start), loss, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0


def category_from_output(output, LABELS):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return LABELS[category_i], category_i


def random_training_example_batch(tensor_x_train, tensor_y_train, n_data_size_train, tensor_x_test, tensor_y_test,
                                  n_data_size_test, batch_size, flag, num=-1):
    if flag == 'train':
        x = tensor_x_train
        y = tensor_y_train
        data_size = n_data_size_train
    elif flag == 'test':
        x = tensor_x_test
        y = tensor_y_test
        data_size = n_data_size_test
    else:
        raise ValueError('flag must be "train" or "test"')

    if num == -1:
        ran_num = random.randint(0, data_size - batch_size)
    else:
        ran_num = num

    pose_sequence_tensor = x[ran_num:(ran_num + batch_size)]
    pose_sequence_tensor = pose_sequence_tensor
    category_tensor = y[ran_num:ran_num + batch_size, :]

    return category_tensor.long(), pose_sequence_tensor


def prepare_data(coordinates_path, n_steps, n_joints):
    x_train = np.empty((0, n_steps, n_joints), dtype='float32')
    x_test = np.empty((0, n_steps, n_joints), dtype='float32')

    y_train = np.empty((0, 1), dtype='int')
    y_test = np.empty((0, 1), dtype='int')

    for subject_id, subject_path in enumerate(sorted([f.path for f in os.scandir(coordinates_path)])):
        for a_id, action_dir_path in enumerate(sorted([f.path for f in os.scandir(subject_path)])):
            for repetition_id, repetition_path in enumerate(sorted([f.path for f in os.scandir(action_dir_path)])):
                x = load_x(os.path.join(repetition_path, 'x.csv'), n_steps)
                if repetition_id < 4:
                    x_train = np.concatenate([x_train, x])
                    y_train = np.concatenate([y_train, np.repeat(a_id, len(x)).reshape((-1, 1))])
                else:
                    x_test = np.concatenate([x_test, x])
                    y_test = np.concatenate([y_test, np.repeat(a_id, len(x)).reshape((-1, 1))])

    return x_train, y_train, x_test, y_test


def load_x(x_path, n_steps):
    file = open(x_path, 'r')
    x = np.array([elem for elem in [row.split(',') for row in file]], dtype=np.float32)
    file.close()
    blocks = int(len(x) / n_steps)
    fixed_size = blocks * n_steps
    return np.array(np.array_split(x[:fixed_size], blocks))


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


if __name__ == '__main__':
    main()
