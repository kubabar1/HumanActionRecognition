import numpy as np
import torch

from .model.LSTMSimpleModel import LSTMSimpleModel
from ...utils.dataset_util import get_analysed_keypoints
from ...utils.evaluation_utils import draw_confusion_matrix


def evaluate_tests(classes, test_data, test_labels, model_path, hidden_size=128, input_size=36):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lstm_model = LSTMSimpleModel(input_size, hidden_size, len(classes)).to(device)
    lstm_model.load_state_dict(torch.load(model_path))
    lstm_model.eval()

    analysed_kpts_left, analysed_kpts_right = get_analysed_keypoints()
    all_analysed_kpts = analysed_kpts_left + analysed_kpts_right

    correct_arr = []
    predicted_arr = []

    for data, label in zip(test_data, test_labels):
        label = np.array(label).reshape(1)
        data = data[:, all_analysed_kpts, :].reshape((1, -1, input_size))

        tensor_val_x = torch.tensor(data, dtype=torch.float, device=device)
        tensor_val_y = torch.from_numpy(label).to(device)

        output_val = lstm_model(tensor_val_x)

        correct = classes[int(tensor_val_y[0])]
        predicted = classes[int(torch.argmax(torch.exp(output_val[0])).item())]

        correct_arr.append(correct)
        predicted_arr.append(predicted)

    draw_confusion_matrix(correct_arr, predicted_arr, classes)


def fit(classes, data, model_path, hidden_size=128, input_size = 36):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lstm_model = LSTMSimpleModel(input_size, hidden_size, len(classes)).to(device)
    lstm_model.load_state_dict(torch.load(model_path))
    lstm_model.eval()

    analysed_kpts_left, analysed_kpts_right = get_analysed_keypoints()
    all_analysed_kpts = analysed_kpts_left + analysed_kpts_right

    data = data[:, all_analysed_kpts, :].reshape((1, -1, input_size))
    tensor_val_x = torch.tensor(data, dtype=torch.float, device=device)
    output_val = lstm_model(tensor_val_x)

    # torch.set_printoptions(precision=5, sci_mode=False)
    # print(torch.exp(output_val))
    # print(classes)

    return classes[int(torch.argmax(torch.exp(output_val[0])).item())]
