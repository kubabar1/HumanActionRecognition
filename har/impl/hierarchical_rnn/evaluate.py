from random import randrange

import numpy as np
import torch

from .model.HierarchicalRNNModel import HierarchicalRNNModel
from .utils.HierarchicalRNNDataset import HierarchicalRNNDataset
from ...utils.dataset_util import DatasetInputType, get_all_body_parts_steps, get_all_body_parts_splits, prepare_body_part_data
from ...utils.evaluation_utils import draw_confusion_matrix


def evaluate_tests(classes, test_data, test_labels, model_path, analysed_kpts_description, hidden_size=128, input_size=9,
                   input_type=DatasetInputType.SPLIT, split=20, steps=32):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hrnn_model = HierarchicalRNNModel(input_size, hidden_size, len(classes)).to(device)
    hrnn_model.load_state_dict(torch.load(model_path))
    hrnn_model.eval()

    test_data_loader = HierarchicalRNNDataset(test_data, test_labels, len(test_data), analysed_kpts_description,
                                              split=split, steps=steps, input_type=input_type, is_test=True)

    test_x, test_y = next(iter(test_data_loader))

    tensor_val_x = torch.tensor(np.array(test_x), dtype=torch.float, device=device)
    tensor_val_y = torch.tensor(np.array(test_y), device=device)

    output_val = hrnn_model(tensor_val_x)

    correct_arr = [classes[int(i)] for i in tensor_val_y]
    predicted_arr = [classes[int(torch.argmax(torch.exp(i)).item())] for i in output_val]

    draw_confusion_matrix(correct_arr, predicted_arr, classes)

    return np.sum([1 for c, p in zip(correct_arr, predicted_arr) if c == p]) / len(correct_arr)


def fit(classes, data, model_path, analysed_kpts_description, hidden_size=128, input_size=9, input_type=DatasetInputType.SPLIT,
        split=20, steps=32):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hrnn_model = HierarchicalRNNModel(input_size, hidden_size, len(classes)).to(device)
    hrnn_model.load_state_dict(torch.load(model_path))
    hrnn_model.eval()

    analysed_body_parts = ['right_wrist', 'left_wrist', 'right_elbow', 'left_elbow', 'right_shoulder', 'left_shoulder',
                           'right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_ankle', 'left_ankle']

    data = prepare_body_part_data(data, analysed_body_parts, analysed_kpts_description, input_type, steps, split)

    tensor_val_x = torch.tensor(np.array(data), dtype=torch.float, device=device)

    output_val = hrnn_model(tensor_val_x)

    return classes[torch.argmax(torch.exp(output_val)).item()]
