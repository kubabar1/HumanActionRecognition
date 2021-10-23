import numpy as np
import torch

from .model.HierarchicalRNNModel import HierarchicalRNNModel
from .utils.HierarchicalRNNDataset import HierarchicalRNNDataset
from ...utils.dataset_util import DatasetInputType, prepare_body_part_data, normalise_skeleton_3d_batch, normalise_skeleton_3d
from ...utils.evaluation_utils import draw_confusion_matrix


def load_model(model_path, classes_count, hidden_size=128, is_3d=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_size = 9 if is_3d else 6
    hrnn_model = HierarchicalRNNModel(input_size, hidden_size, classes_count).to(device)
    hrnn_model.load_state_dict(torch.load(model_path))
    hrnn_model.eval()
    return hrnn_model


def evaluate_tests(classes, test_data, test_labels, hrnn_model, analysed_kpts_description, input_type=DatasetInputType.SPLIT, split=20,
                   steps=32, show_diagram=True, results_path='results', use_normalization=True):
    if use_normalization:
        test_data = normalise_skeleton_3d_batch(test_data, analysed_kpts_description['left_hip'], analysed_kpts_description['right_hip'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_data_loader = HierarchicalRNNDataset(test_data, test_labels, len(test_data), analysed_kpts_description,
                                              split=split, steps=steps, input_type=input_type, is_test=True)

    test_x, test_y = next(iter(test_data_loader))

    tensor_val_x = torch.tensor(np.array(test_x), dtype=torch.float, device=device)
    tensor_val_y = torch.tensor(np.array(test_y), device=device)

    output_val = hrnn_model(tensor_val_x)

    correct_arr = [classes[int(i)] for i in tensor_val_y]
    predicted_arr = [classes[int(torch.argmax(torch.exp(i)).item())] for i in output_val]

    draw_confusion_matrix(correct_arr, predicted_arr, classes, show_diagram=show_diagram, result_path=results_path)

    return np.sum([1 for c, p in zip(correct_arr, predicted_arr) if c == p]) / len(correct_arr)


def fit(classes, data, hrnn_model, analysed_kpts_description, input_type=DatasetInputType.SPLIT, split=20, steps=32,
        use_normalization=True):
    if use_normalization:
        data = normalise_skeleton_3d(data, analysed_kpts_description['left_hip'], analysed_kpts_description['right_hip'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    analysed_body_parts = ['right_wrist', 'left_wrist', 'right_elbow', 'left_elbow', 'right_shoulder', 'left_shoulder',
                           'right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_ankle', 'left_ankle']

    data = prepare_body_part_data(data, analysed_body_parts, analysed_kpts_description, input_type, steps, split)

    tensor_val_x = torch.tensor(np.array(data), dtype=torch.float, device=device)

    output_val = hrnn_model(tensor_val_x)

    return classes[torch.argmax(torch.exp(output_val)).item()]
