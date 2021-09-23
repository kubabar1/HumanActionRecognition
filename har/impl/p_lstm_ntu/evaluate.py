import warnings

import numpy as np
import torch

from .model.PLSTMModel import PLSTMModel
from .utils.PLSTMDataset import PLSTMDataset
from ...utils.dataset_util import DatasetInputType, get_all_body_parts_splits, get_all_body_parts_steps
from ...utils.evaluation_utils import draw_confusion_matrix


def load_model(model_path, classes_count, hidden_size=128, is_3d=True):
    warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_size = 9 if is_3d else 6
    p_lstm_model = PLSTMModel(input_size, hidden_size, classes_count).to(device)
    p_lstm_model.load_state_dict(torch.load(model_path))
    p_lstm_model.eval()
    return p_lstm_model


def evaluate_tests(classes, test_data, test_labels, p_lstm_model, analysed_kpts_description, input_type=DatasetInputType.SPLIT, steps=32,
                   split=20, show_diagram=True, results_path='results'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_data_loader = PLSTMDataset(test_data, test_labels, len(test_data), analysed_kpts_description, steps=steps, split=split,
                                    input_type=input_type, is_test=True)

    test_x, test_y = next(iter(test_data_loader))

    tensor_val_x = torch.tensor(np.array(test_x), dtype=torch.float, device=device)
    tensor_val_y = torch.tensor(np.array(test_y), device=device)

    output_val = p_lstm_model(tensor_val_x)

    correct_arr = [classes[int(i)] for i in tensor_val_y]
    predicted_arr = [classes[int(torch.argmax(torch.exp(i)).item())] for i in output_val]

    draw_confusion_matrix(correct_arr, predicted_arr, classes, result_path=results_path, show_diagram=show_diagram)

    return np.sum([1 for c, p in zip(correct_arr, predicted_arr) if c == p]) / len(correct_arr)


def fit(classes, data, p_lstm_model, analysed_kpts_description, input_type=DatasetInputType.SPLIT, steps=32, split=20):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    analysed_body_parts = ['right_wrist', 'left_wrist', 'right_elbow', 'left_elbow', 'right_shoulder', 'left_shoulder',
                           'right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_ankle', 'left_ankle']

    data = prepare_data(data, analysed_body_parts, analysed_kpts_description, input_type, steps, split)

    tensor_val_x = torch.tensor(np.array(data), dtype=torch.float, device=device)

    output_val = p_lstm_model(tensor_val_x)

    return classes[torch.argmax(torch.exp(output_val)).item()]


def prepare_data(data, analysed_body_parts, analysed_kpts_description, input_type, steps, split):
    right_wrists = []
    left_wrists = []
    right_elbows = []
    left_elbows = []
    right_shoulders = []
    left_shoulders = []
    right_hips = []
    left_hips = []
    right_knees = []
    left_knees = []
    right_ankles = []
    left_ankles = []

    if input_type == DatasetInputType.STEP:
        parts = int(data.shape[0] / steps)
        for i in range(parts):
            begin = i * steps
            end = i * steps + steps
            body_el = get_all_body_parts_steps(data, analysed_body_parts, analysed_kpts_description, begin, end)
            right_wrists.append(body_el['right_wrist'])
            left_wrists.append(body_el['left_wrist'])
            right_elbows.append(body_el['right_elbow'])
            left_elbows.append(body_el['left_elbow'])
            right_shoulders.append(body_el['right_shoulder'])
            left_shoulders.append(body_el['left_shoulder'])
            right_hips.append(body_el['right_hip'])
            left_hips.append(body_el['left_hip'])
            right_knees.append(body_el['right_knee'])
            left_knees.append(body_el['left_knee'])
            right_ankles.append(body_el['right_ankle'])
            left_ankles.append(body_el['left_ankle'])
    elif input_type == DatasetInputType.SPLIT:
        body_el = get_all_body_parts_splits(data, analysed_body_parts, analysed_kpts_description, split)
        right_wrists.append(body_el['right_wrist'])
        left_wrists.append(body_el['left_wrist'])
        right_elbows.append(body_el['right_elbow'])
        left_elbows.append(body_el['left_elbow'])
        right_shoulders.append(body_el['right_shoulder'])
        left_shoulders.append(body_el['left_shoulder'])
        right_hips.append(body_el['right_hip'])
        left_hips.append(body_el['left_hip'])
        right_knees.append(body_el['right_knee'])
        left_knees.append(body_el['left_knee'])
        right_ankles.append(body_el['right_ankle'])
        left_ankles.append(body_el['left_ankle'])
    else:
        raise ValueError('Invalid or unimplemented input type')

    left_arms = np.concatenate((left_wrists, left_elbows, left_shoulders), axis=2)
    right_arms = np.concatenate((right_wrists, right_elbows, right_shoulders), axis=2)
    left_legs = np.concatenate((left_hips, left_knees, left_ankles), axis=2)
    right_legs = np.concatenate((right_hips, right_knees, right_ankles), axis=2)

    return [left_arms, right_arms, left_legs, right_legs]
