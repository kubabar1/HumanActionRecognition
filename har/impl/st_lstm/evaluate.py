from random import randrange

import numpy as np
import torch

from .model.STLSTMModel import STLSTMModel
from .utils.STLSTMDataset import STLSTMDataset
from ...utils.dataset_util import DatasetInputType
from ...utils.evaluation_utils import draw_confusion_matrix


def evaluate_tests(classes, test_data, test_labels, model_path, analysed_kpts_description, hidden_size=128,
                   input_size=3, input_type=DatasetInputType.SPLIT, split=20, steps=32, dropout=0.5, joints_count=12,
                   use_tau=False, use_bias=True, use_two_layers=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    st_lstm_model = STLSTMModel(input_size, joints_count, hidden_size, len(classes), dropout, use_tau=use_tau,
                                bias=use_bias, use_two_layers=use_two_layers).to(device)
    st_lstm_model.load_state_dict(torch.load(model_path))
    st_lstm_model.eval()

    test_data_loader = STLSTMDataset(test_data, test_labels, len(test_data), analysed_kpts_description, steps=steps, split=split,
                                     input_type=input_type, is_test=True)

    test_x, test_y = next(iter(test_data_loader))

    tensor_val_x = torch.tensor(test_x, dtype=torch.float, device=device)
    tensor_val_y = torch.from_numpy(test_y).to(device)

    output_val = st_lstm_model(tensor_val_x)

    correct_arr = [classes[int(i)] for i in tensor_val_y]
    predicted_arr = [classes[int(torch.argmax(torch.exp(i)).item())] for i in output_val]

    draw_confusion_matrix(correct_arr, predicted_arr, classes)

    return np.sum([1 for c, p in zip(correct_arr, predicted_arr) if c == p]) / len(correct_arr)


def fit(classes, data, model_path, analysed_kpts_description, hidden_size=128,
        input_size=3, input_type=DatasetInputType.SPLIT, split=20, dropout=0.5, joints_count=12,
        use_tau=False, use_bias=True, use_two_layers=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    st_lstm_model = STLSTMModel(input_size, joints_count, hidden_size, len(classes), dropout, use_tau=use_tau,
                                bias=use_bias, use_two_layers=use_two_layers).to(device)
    st_lstm_model.load_state_dict(torch.load(model_path))
    st_lstm_model.eval()

    all_analysed_kpts = list(analysed_kpts_description.values())

    data = np.array([data[:, all_analysed_kpts, :]])

    if input_type == DatasetInputType.SPLIT:
        data = np.array([a[:, randrange(len(a)), :, :] for a in np.array_split(data, split, axis=1)])
        data = data.reshape((1, split, len(all_analysed_kpts), input_size))

    tensor_val_x = torch.tensor(data, dtype=torch.float, device=device)
    output_val = st_lstm_model(tensor_val_x)

    # torch.set_printoptions(precision=5, sci_mode=False)
    # print(torch.exp(output_val))
    # print(classes)

    return classes[int(torch.argmax(torch.exp(output_val[0])).item())]
