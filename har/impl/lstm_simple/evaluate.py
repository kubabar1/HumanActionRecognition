from random import randrange

import numpy as np
import torch

from .model.LSTMSimpleModel import LSTMSimpleModel
from .utils.LSTMSimpleDataset import LSTMSimpleDataset
from ...utils.dataset_util import DatasetInputType, SetType
from ...utils.evaluation_utils import draw_confusion_matrix


def load_model(model_path, classes, analysed_kpts_count=12, hidden_size=128, hidden_layers=3, is_3d=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_size = analysed_kpts_count * (3 if is_3d else 2)
    lstm_model = LSTMSimpleModel(input_size, hidden_size, hidden_layers, len(classes)).to(device)
    lstm_model.load_state_dict(torch.load(model_path))
    lstm_model.eval()
    return lstm_model


def evaluate_tests(classes, test_data, test_labels, lstm_simple_model, analysed_kpts_description, input_type=DatasetInputType.SPLIT,
                   split=20, steps=32, show_diagram=True, results_path='results'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_data_loader = LSTMSimpleDataset(test_data, test_labels, len(test_data), analysed_kpts_description, steps=steps,
                                         split=split, input_type=input_type, is_test=True, set_type=SetType.TEST)

    test_x, test_y = next(iter(test_data_loader))

    tensor_val_x = torch.tensor(test_x.reshape((test_x.shape[0], test_x.shape[1], -1)), dtype=torch.float, device=device)
    tensor_val_y = torch.tensor(np.array(test_y), device=device)

    output_val = lstm_simple_model(tensor_val_x)

    correct_arr = [classes[int(i)] for i in tensor_val_y]
    predicted_arr = [classes[int(torch.argmax(torch.exp(i)).item())] for i in output_val]

    draw_confusion_matrix(correct_arr, predicted_arr, classes, show_diagram=show_diagram, result_path=results_path)

    return np.sum([1 for c, p in zip(correct_arr, predicted_arr) if c == p]) / len(correct_arr)


def fit(classes, data, lstm_simple_model, analysed_kpts_description, input_type=DatasetInputType.SPLIT, split=20):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_analysed_kpts = list(analysed_kpts_description.values())
    input_size = len(all_analysed_kpts) * data.shape[2]
    data = data[:, all_analysed_kpts, :].reshape((1, -1, input_size))

    if input_type == DatasetInputType.SPLIT:
        data = np.array([a[:, randrange(len(a)), :] for a in np.array_split(data, split, axis=1)])
        data = data.reshape((1, -1, input_size))

    tensor_val_x = torch.tensor(data, dtype=torch.float, device=device)
    output_val = lstm_simple_model(tensor_val_x)

    return classes[int(torch.argmax(torch.exp(output_val[0])).item())]
