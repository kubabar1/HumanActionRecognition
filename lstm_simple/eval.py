import torch
import numpy as np
import os

from LSTM import LSTM


def load_x(x_path, n_steps):
    file = open(x_path, 'r')
    x = np.array([elem for elem in [row.split(',') for row in file]], dtype=np.float32)
    file.close()
    blocks = int(len(x) / n_steps)
    fixed_size = blocks * n_steps
    return np.array(np.array_split(x[:fixed_size], blocks))


def category_from_output(output, LABELS):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return LABELS[category_i], category_i


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
n_hidden = 128
n_layer = 3
n_steps = 32
n_joints = 34
n_categories = len(LABELS)

pp = '/home/kuba/workspace/human_action_recognition/HumanActionRecognition/lstm_simple/data/coordinates/s01/a10/r05/x.csv'
x_test = load_x(pp, n_steps)
tensor_x_test = torch.from_numpy(x_test)

model = LSTM(n_joints, n_hidden, n_categories, n_layer)
model.load_state_dict(torch.load('./model.pth'))
model.eval()
output = model(tensor_x_test)
guess, guess_i = category_from_output(output, LABELS)
print(guess)
print(guess_i)
