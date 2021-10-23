from enum import Enum

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from .utils.JTMDataset import JTMDataset, generate_sample_images
from ...utils.dataset_util import SetType, normalise_skeleton_3d_batch, normalise_skeleton_3d
from ...utils.evaluation_utils import draw_confusion_matrix


class ModelType(Enum):
    FRONT = 0
    TOP = 1
    SIDE = 2


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


def load_model(model_path, classes_count):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    jtm_model = models.AlexNet()
    jtm_model.classifier[6] = nn.Linear(4096, classes_count)
    jtm_model.load_state_dict(torch.load(model_path))
    jtm_model.eval()
    jtm_model.to(device)
    return jtm_model


def evaluate_tests_min(classes, test_data, test_labels, jtm_model, analysed_kpts_description, image_width, image_height,
                       model_type: ModelType, result_path='results', show_diagram=True, use_normalization=True):
    if use_normalization:
        test_data = normalise_skeleton_3d_batch(test_data, analysed_kpts_description['left_hip'], analysed_kpts_description['right_hip'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_data_loader = JTMDataset(test_data, test_labels, image_width, image_height, len(test_data),
                                  SetType.TEST, analysed_kpts_description, is_test=True)

    test_x, test_y = next(iter(test_data_loader))

    test_acc = 0
    pred = []

    for i, (test_img, test_lbl) in enumerate(zip(test_x, test_y)):
        test_img_tensor = torch.unsqueeze(transform(test_img[model_type.value]), 0).to(device)

        test_output = torch.softmax(jtm_model(test_img_tensor), 1)
        pred.append(torch.argmax(test_output).item())

        if torch.argmax(test_output).item() == test_lbl:
            test_acc += 1

    correct_arr = [classes[int(i)] for i in test_y]
    predicted_arr = [classes[int(i)] for i in pred]

    draw_confusion_matrix(correct_arr, predicted_arr, classes, result_path=result_path, show_diagram=show_diagram)

    return test_acc / len(test_x)


def evaluate_tests(classes, test_data, test_labels, jtm_model_front, jtm_model_top, jtm_model_side, analysed_kpts_description,
                   image_width, image_height, result_path='results', show_diagram=True, use_normalization=True):
    if use_normalization:
        test_data = normalise_skeleton_3d_batch(test_data, analysed_kpts_description['left_hip'], analysed_kpts_description['right_hip'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_data_loader = JTMDataset(test_data, test_labels, image_width, image_height, len(test_data),
                                  SetType.TEST, analysed_kpts_description, is_test=True)

    test_x, test_y = next(iter(test_data_loader))

    test_acc = 0
    pred = []

    for i, (test_img, test_lbl) in enumerate(zip(test_x, test_y)):
        test_img_tensor_front = torch.unsqueeze(transform(test_img[ModelType.FRONT.value]), 0).to(device)
        test_img_tensor_top = torch.unsqueeze(transform(test_img[ModelType.TOP.value]), 0).to(device)
        test_img_tensor_side = torch.unsqueeze(transform(test_img[ModelType.SIDE.value]), 0).to(device)

        test_output_front = torch.softmax(jtm_model_front(test_img_tensor_front), 1)
        test_output_top = torch.softmax(jtm_model_top(test_img_tensor_top), 1)
        test_output_side = torch.softmax(jtm_model_side(test_img_tensor_side), 1)

        test_output = test_output_front * test_output_top * test_output_side

        pred.append(torch.argmax(test_output).item())

        if torch.argmax(test_output).item() == test_lbl:
            test_acc += 1

    correct_arr = [classes[int(i)] for i in test_y]
    predicted_arr = [classes[int(i)] for i in pred]

    draw_confusion_matrix(correct_arr, predicted_arr, classes, result_path=result_path, show_diagram=show_diagram)

    return test_acc / len(test_x)


def fit_min(classes, data, jtm_model, analysed_kpts_description, image_width, image_height, model_type: ModelType, use_normalization=True):
    if use_normalization:
        data = normalise_skeleton_3d(data, analysed_kpts_description['left_hip'], analysed_kpts_description['right_hip'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    smpl_img = generate_sample_images(data, analysed_kpts_description, image_width, image_height)[model_type.value]

    test_img_tensor = torch.unsqueeze(transform(smpl_img), 0).to(device)
    test_output = torch.softmax(jtm_model(test_img_tensor), 1)

    return classes[torch.argmax(test_output).item()]


def fit(classes, data, jtm_model_front, jtm_model_top, jtm_model_side, analysed_kpts_description, image_width, image_height,
        use_normalization=True):
    if use_normalization:
        data = normalise_skeleton_3d(data, analysed_kpts_description['left_hip'], analysed_kpts_description['right_hip'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    smpl_img_front = generate_sample_images(data, analysed_kpts_description, image_width, image_height)[ModelType.FRONT.value]
    smpl_img_top = generate_sample_images(data, analysed_kpts_description, image_width, image_height)[ModelType.TOP.value]
    smpl_img_side = generate_sample_images(data, analysed_kpts_description, image_width, image_height)[ModelType.SIDE.value]

    test_img_tensor_front = torch.unsqueeze(transform(smpl_img_front), 0).to(device)
    test_img_tensor_top = torch.unsqueeze(transform(smpl_img_top), 0).to(device)
    test_img_tensor_side = torch.unsqueeze(transform(smpl_img_side), 0).to(device)

    test_output_front = torch.softmax(jtm_model_front(test_img_tensor_front), 1)
    test_output_top = torch.softmax(jtm_model_top(test_img_tensor_top), 1)
    test_output_side = torch.softmax(jtm_model_side(test_img_tensor_side), 1)

    test_output = test_output_front * test_output_top * test_output_side

    return classes[torch.argmax(test_output).item()]
